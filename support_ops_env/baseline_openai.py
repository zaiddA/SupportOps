# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI baseline runner for the bundled tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .models import ActionType, Priority, SupportOpsAction, Team, TicketStatus
from .tasks import TASK_ORDER
from .trainer_env import SupportOpsTrainerEnv


DEFAULT_MODEL = "gpt-4.1-mini"

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
        "tags_to_add": {
            "type": "array",
            "items": {"type": "string"},
        },
        "tags_to_remove": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resolution_code": {"type": ["string", "null"]},
        "response_text": {"type": ["string", "null"]},
        "note_text": {"type": ["string", "null"]},
        "target_ticket_id": {"type": ["string", "null"]},
    },
    "required": ["action_type", "ticket_id", "tags_to_add", "tags_to_remove"],
}


def main() -> None:
    """Run the selected model on one or more tasks and print a JSON summary."""
    parser = argparse.ArgumentParser(description="Run an OpenAI baseline on Support Ops Env.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use.")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=TASK_ORDER,
        help="Task ids to run. Defaults to all bundled tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the baseline summary as JSON.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run the OpenAI baseline.")

    client = OpenAI()
    summary = {
        "model": args.model,
        "tasks": [],
    }

    for task_id in args.tasks:
        result = run_single_task(client=client, model=args.model, task_id=task_id)
        summary["tasks"].append(result)

    if summary["tasks"]:
        summary["average_score"] = round(
            sum(item["final_score"] for item in summary["tasks"]) / len(summary["tasks"]),
            4,
        )
    else:
        summary["average_score"] = 0.0

    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_single_task(client: OpenAI, model: str, task_id: str) -> dict:
    """Execute one full episode with the selected model."""
    env = SupportOpsTrainerEnv(task_id=task_id)
    observation = env.reset(task_id=task_id)
    trajectory: list[dict[str, object]] = []
    total_reward = 0.0

    done = False
    while not done:
        action = plan_next_action(client=client, model=model, observation=observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        trajectory.append(
            {
                "action": action.model_dump(mode="json"),
                "reward": round(reward, 4),
                "done": done,
                "score": info.get("score"),
                "last_action_result": observation.last_action_result,
            }
        )

    final_state = env.state()
    return {
        "task_id": task_id,
        "final_score": round(final_state.current_score, 4),
        "total_reward": round(total_reward, 4),
        "steps": final_state.step_count,
        "trajectory": trajectory,
    }


def plan_next_action(client: OpenAI, model: str, observation) -> SupportOpsAction:
    """Ask the model for the next action under a strict JSON schema."""
    prompt = build_turn_prompt(observation)
    response = client.responses.create(
        model=model,
        temperature=0,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        max_output_tokens=450,
        text={
            "format": {
                "type": "json_schema",
                "name": "support_ops_action",
                "schema": ACTION_SCHEMA,
                "strict": True,
            }
        },
    )
    payload = json.loads(response.output_text)
    return SupportOpsAction(**payload)


def build_turn_prompt(observation) -> str:
    """Create the model prompt from the current observation."""
    visible_observation = observation.model_dump(
        mode="json",
        exclude={"done", "reward", "metadata", "reward_breakdown"},
    )
    return (
        "Choose the single best next action for this support workflow state.\n"
        "Observation JSON:\n"
        f"{json.dumps(visible_observation, indent=2)}\n"
        "Remember: return only the structured action."
    )


if __name__ == "__main__":
    main()
