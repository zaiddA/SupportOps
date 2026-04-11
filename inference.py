#!/usr/bin/env python3

"""Submission-style inference runner with structured START/STEP/END logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from .client import SupportOpsEnv
    from .models import ActionType, Priority, SupportOpsAction, Team, TicketStatus
    from .tasks import TASK_ORDER
except ImportError:
    from client import SupportOpsEnv
    from models import ActionType, Priority, SupportOpsAction, Team, TicketStatus
    from tasks import TASK_ORDER


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://zaidd1-support-ops-env.hf.space")
BENCHMARK_NAME = "support_ops_env"
DEFAULT_MAX_STEPS = 12

SYSTEM_PROMPT = """
You are an operations agent working in a B2B SaaS support inbox.

Rules:
- Return exactly one JSON action that matches the schema.
- Only use ticket ids that appear in the observation.
- Be conservative and policy-driven.
- Keep customer replies concise, empathetic, and factual.
- Do not edit unrelated tickets.
- Finish only when the inbox state already satisfies the objective.
""".strip()

ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [action.value for action in ActionType],
        },
        "ticket_id": {"type": ["string", "null"]},
        "status": {
            "type": ["string", "null"],
            "enum": [status.value for status in TicketStatus] + [None],
        },
        "priority": {
            "type": ["string", "null"],
            "enum": [priority.value for priority in Priority] + [None],
        },
        "assigned_team": {
            "type": ["string", "null"],
            "enum": [team.value for team in Team] + [None],
        },
        "tags_to_add": {"type": "array", "items": {"type": "string"}},
        "tags_to_remove": {"type": "array", "items": {"type": "string"}},
        "resolution_code": {"type": ["string", "null"]},
        "response_text": {"type": ["string", "null"]},
        "note_text": {"type": ["string", "null"]},
        "target_ticket_id": {"type": ["string", "null"]},
    },
    "required": ["action_type", "ticket_id", "tags_to_add", "tags_to_remove"],
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = sanitize_log_value(error) if error else "null"
    done_value = str(done).lower()
    action_value = sanitize_log_value(action)
    print(
        f"[STEP] step={step} action={action_value} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_value = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_value}",
        flush=True,
    )


def sanitize_log_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).replace("=", ":")[:160] or "null"


def create_llm_client() -> OpenAI:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set HF_TOKEN or OPENAI_API_KEY before running inference.py.")
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


async def create_env_client() -> SupportOpsEnv:
    if LOCAL_IMAGE_NAME:
        return await SupportOpsEnv.from_docker_image(LOCAL_IMAGE_NAME)
    env = SupportOpsEnv(base_url=ENV_BASE_URL)
    await env.connect()
    return env


def build_turn_prompt(observation: Any) -> str:
    visible_observation = observation.model_dump(
        mode="json",
        exclude={"done", "reward", "metadata", "reward_breakdown"},
    )
    return (
        "Choose the single best next action for this support workflow state.\n"
        "Observation JSON:\n"
        f"{json.dumps(visible_observation, indent=2)}\n"
        "Return only one valid JSON action."
    )


def parse_json_action(raw_text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(.*?)\s*```", raw_text, flags=re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if object_match:
        return json.loads(object_match.group(0))

    raise ValueError("Model response did not contain a valid JSON object.")


def finish_action() -> SupportOpsAction:
    return SupportOpsAction(
        action_type=ActionType.FINISH,
        ticket_id=None,
        tags_to_add=[],
        tags_to_remove=[],
    )


def plan_next_action(client: OpenAI, observation: Any) -> SupportOpsAction:
    prompt = build_turn_prompt(observation)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=450,
    )
    raw_text = completion.choices[0].message.content or "{}"
    payload = parse_json_action(raw_text)
    return SupportOpsAction(**payload)


def action_summary(action: SupportOpsAction) -> str:
    compact = {
        "action_type": action.action_type.value,
        "ticket_id": action.ticket_id,
        "status": action.status.value if action.status else None,
        "priority": action.priority.value if action.priority else None,
        "assigned_team": action.assigned_team.value if action.assigned_team else None,
        "resolution_code": action.resolution_code,
        "target_ticket_id": action.target_ticket_id,
    }
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=True)


async def run_task(env: SupportOpsEnv, client: OpenAI, task_id: str) -> dict[str, Any]:
    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    reset_result = await env.reset(task_id=task_id)
    observation = reset_result.observation
    rewards: list[float] = []
    trajectory: list[dict[str, Any]] = []
    done = bool(reset_result.done)
    step_index = 0
    max_steps = int(getattr(observation, "remaining_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)

    while not done and step_index < max_steps:
        step_index += 1
        error_message: str | None = None
        try:
            action = plan_next_action(client=client, observation=observation)
        except Exception as exc:
            error_message = f"model_error:{exc}"
            action = finish_action()

        action_label = action_summary(action)
        step_result = await env.step(action)
        observation = step_result.observation
        reward = float(step_result.reward or 0.0)
        done = bool(step_result.done)
        warning_text = "; ".join(observation.warnings) if observation.warnings else None
        combined_error = "; ".join(part for part in [error_message, warning_text] if part) or None
        rewards.append(reward)
        trajectory.append(
            {
                "step": step_index,
                "action": action.model_dump(mode="json"),
                "reward": round(reward, 4),
                "done": done,
                "warnings": observation.warnings,
            }
        )
        log_step(
            step=step_index,
            action=action_label,
            reward=reward,
            done=done,
            error=combined_error,
        )

    final_state = await env.state()
    final_score = round(final_state.current_score, 4)
    success = final_score >= 0.95
    log_end(success=success, steps=step_index, score=final_score, rewards=rewards)
    return {
        "task_id": task_id,
        "success": success,
        "score": final_score,
        "steps": step_index,
        "rewards": [round(reward, 4) for reward in rewards],
        "trajectory": trajectory,
    }


async def run_inference(task_ids: list[str], output_path: Path | None) -> int:
    client = create_llm_client()
    env = await create_env_client()
    summary = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "env_base_url": ENV_BASE_URL if not LOCAL_IMAGE_NAME else "docker_image",
        "tasks": [],
    }

    try:
        for task_id in task_ids:
            summary["tasks"].append(await run_task(env=env, client=client, task_id=task_id))
    finally:
        await env.close()

    if summary["tasks"]:
        summary["average_score"] = round(
            sum(item["score"] for item in summary["tasks"]) / len(summary["tasks"]),
            4,
        )
    else:
        summary["average_score"] = 0.0

    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run submission-style inference on Support Ops Env.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=TASK_ORDER,
        help="Task ids to evaluate. Defaults to all bundled tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for the aggregate summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(asyncio.run(run_inference(task_ids=args.tasks, output_path=args.output)))
    except KeyboardInterrupt:
        print("Inference interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
