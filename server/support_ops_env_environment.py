# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core environment logic for the support inbox simulation."""

from __future__ import annotations

import json
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..graders import TASK_SCORE_EPSILON, TaskGrade, grade_task
    from ..models import (
        ActionRecord,
        ActionType,
        Priority,
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsReward,
        SupportOpsState,
        Team,
        TicketRecord,
        TicketStatus,
    )
    from ..tasks import EASY_TASK_ID, TASK_ORDER, TaskDefinition, build_task_catalog
except ImportError:
    from graders import TASK_SCORE_EPSILON, TaskGrade, grade_task
    from models import (
        ActionRecord,
        ActionType,
        Priority,
        SupportOpsAction,
        SupportOpsObservation,
        SupportOpsReward,
        SupportOpsState,
        Team,
        TicketRecord,
        TicketStatus,
    )
    from tasks import EASY_TASK_ID, TASK_ORDER, TaskDefinition, build_task_catalog


class SupportOpsEnvironment(Environment):
    """Stateful ticket-workflow environment with deterministic grading."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task_id: str | None = None):
        self._task_catalog = build_task_catalog()
        self._default_task_id = default_task_id or EASY_TASK_ID
        self._state: SupportOpsState | None = None
        self._current_task: TaskDefinition | None = None
        self._tickets: dict[str, TicketRecord] = {}
        self._initial_tickets: dict[str, TicketRecord] = {}
        self._current_grade = TaskGrade(score=TASK_SCORE_EPSILON, criteria=[])
        self._action_signatures: dict[str, int] = {}
        self._action_history: list[ActionRecord] = []
        self._last_action_result = "Environment initialized."
        self._last_reward_breakdown: SupportOpsReward | None = None
        self._done = False
        self.reset(task_id=self._default_task_id)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> SupportOpsObservation:
        """Start a fresh episode for the selected task."""

        selected_task_id = kwargs.get("task_id")
        if selected_task_id is None:
            if seed is None:
                selected_task_id = self._default_task_id
            else:
                selected_task_id = TASK_ORDER[seed % len(TASK_ORDER)]
        if selected_task_id not in self._task_catalog:
            raise ValueError(f"Unknown task_id {selected_task_id!r}. Expected one of {TASK_ORDER}.")

        self._current_task = self._task_catalog[selected_task_id]
        self._tickets = {
            ticket.ticket_id: ticket.model_copy(deep=True)
            for ticket in self._current_task.tickets
        }
        self._initial_tickets = {
            ticket_id: ticket.model_copy(deep=True)
            for ticket_id, ticket in self._tickets.items()
        }
        self._action_signatures = {}
        self._action_history = []
        self._last_reward_breakdown = None
        self._last_action_result = f"Loaded task {self._current_task.title}."
        self._done = False
        self._current_grade = grade_task(
            self._current_task,
            self._tickets,
            self._initial_tickets,
        )
        self._state = self._build_state(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return self._build_observation(reward=0.0, warnings=[])

    def step(
        self,
        action: SupportOpsAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> SupportOpsObservation:
        """Apply one action and return the next observation."""

        del timeout_s, kwargs
        if self._state is None or self._current_task is None:
            return self.reset()

        if self._done:
            reward = SupportOpsReward(
                total=-0.1,
                score_before=self._current_grade.score,
                score_after=self._current_grade.score,
                score_delta=0.0,
                action_cost=0.0,
                invalid_action_penalty=0.1,
                loop_penalty=0.0,
                finish_bonus=0.0,
                notes=["Episode already finished."],
            )
            self._last_reward_breakdown = reward
            return self._build_observation(reward=reward.total, warnings=["Episode already finished."])

        self._state.step_count += 1
        score_before = self._current_grade.score
        warnings: list[str] = []
        invalid_penalty = 0.0
        finish_bonus = 0.0
        action_cost = 0.01

        signature = json.dumps(
            action.model_dump(mode="json", exclude={"metadata"}),
            sort_keys=True,
        )
        repeat_count = self._action_signatures.get(signature, 0)
        self._action_signatures[signature] = repeat_count + 1
        loop_penalty = 0.03 * repeat_count if repeat_count else 0.0
        if repeat_count:
            warnings.append("Repeated the same action signature; loop penalty applied.")

        summary, penalty_delta, extra_warnings = self._apply_action(action)
        invalid_penalty += penalty_delta
        warnings.extend(extra_warnings)
        self._last_action_result = summary

        self._current_grade = grade_task(
            self._current_task,
            self._tickets,
            self._initial_tickets,
        )
        score_after = self._current_grade.score

        if action.action_type == ActionType.FINISH:
            self._done = True
            finish_bonus = 0.05 if score_after >= 0.95 else -0.1
            warnings.append(
                "Episode finished by agent."
                if score_after >= 0.95
                else "Finished before meeting the full objective."
            )
        elif self._state.step_count >= self._current_task.max_steps:
            self._done = True
            warnings.append("Maximum step limit reached.")

        total_reward = (score_after - score_before) - action_cost - invalid_penalty - loop_penalty + finish_bonus
        reward = SupportOpsReward(
            total=round(total_reward, 4),
            score_before=round(score_before, 4),
            score_after=round(score_after, 4),
            score_delta=round(score_after - score_before, 4),
            action_cost=round(action_cost, 4),
            invalid_action_penalty=round(invalid_penalty, 4),
            loop_penalty=round(loop_penalty, 4),
            finish_bonus=round(finish_bonus, 4),
            notes=warnings.copy(),
        )
        self._last_reward_breakdown = reward
        self._action_history.append(
            ActionRecord(
                step_index=self._state.step_count,
                action_type=action.action_type,
                ticket_id=action.ticket_id,
                summary=summary,
                reward=reward,
                score_after=round(score_after, 4),
            )
        )
        self._state = self._build_state(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )
        return self._build_observation(reward=reward.total, warnings=warnings)

    @property
    def state(self) -> SupportOpsState:
        """Return the current internal state."""

        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    def _apply_action(self, action: SupportOpsAction) -> tuple[str, float, list[str]]:
        warnings: list[str] = []
        invalid_penalty = 0.0

        if action.action_type == ActionType.FINISH:
            return "Marked task complete.", 0.0, warnings

        ticket, error = self._require_ticket(action.ticket_id)
        if ticket is None:
            return error, 0.1, [error]

        if ticket.merged_into:
            warning = f"{ticket.ticket_id} has already been merged into {ticket.merged_into}."
            return warning, 0.08, [warning]

        if action.action_type == ActionType.UPDATE_TICKET:
            changes = []
            if action.status is not None and ticket.status != action.status:
                ticket.status = action.status
                changes.append(f"status={action.status.value}")
            if action.priority is not None and ticket.priority != action.priority:
                ticket.priority = action.priority
                changes.append(f"priority={action.priority.value}")
            if action.assigned_team is not None and ticket.assigned_team != action.assigned_team:
                ticket.assigned_team = action.assigned_team
                changes.append(f"assigned_team={action.assigned_team.value}")
            if action.resolution_code is not None and ticket.resolution_code != action.resolution_code:
                ticket.resolution_code = action.resolution_code
                changes.append(f"resolution_code={action.resolution_code}")
            if action.tags_to_add or action.tags_to_remove:
                current_tags = set(ticket.tags)
                current_tags.update(tag for tag in action.tags_to_add if tag)
                current_tags.difference_update(tag for tag in action.tags_to_remove if tag)
                new_tags = sorted(current_tags)
                if new_tags != sorted(ticket.tags):
                    ticket.tags = new_tags
                    changes.append(f"tags={new_tags}")
            if not changes:
                return f"No ticket fields changed for {ticket.ticket_id}.", 0.05, [
                    "Update action did not modify the ticket.",
                ]
            return f"Updated {ticket.ticket_id}: {', '.join(changes)}.", invalid_penalty, warnings

        if action.action_type == ActionType.REPLY_TO_CUSTOMER:
            response_text = (action.response_text or "").strip()
            if not response_text:
                return f"No reply text supplied for {ticket.ticket_id}.", 0.08, [
                    "reply_to_customer requires response_text.",
                ]
            ticket.replies.append(response_text)
            return f"Sent customer reply on {ticket.ticket_id}.", invalid_penalty, warnings

        if action.action_type == ActionType.ADD_INTERNAL_NOTE:
            note_text = (action.note_text or "").strip()
            if not note_text:
                return f"No note text supplied for {ticket.ticket_id}.", 0.08, [
                    "add_internal_note requires note_text.",
                ]
            ticket.internal_notes.append(note_text)
            return f"Added internal note on {ticket.ticket_id}.", invalid_penalty, warnings

        if action.action_type == ActionType.MERGE_TICKET:
            target, target_error = self._require_ticket(action.target_ticket_id)
            if target is None:
                return target_error, 0.1, [target_error]
            if target.ticket_id == ticket.ticket_id:
                return "Cannot merge a ticket into itself.", 0.1, ["Invalid merge target."]
            if target.company != ticket.company:
                return (
                    "Duplicate merges are only allowed within the same company.",
                    0.12,
                    ["Cross-company merge blocked."],
                )
            ticket.merged_into = target.ticket_id
            ticket.status = TicketStatus.CLOSED_DUPLICATE
            current_tags = set(ticket.tags)
            current_tags.add("duplicate")
            ticket.tags = sorted(current_tags)
            return f"Merged {ticket.ticket_id} into {target.ticket_id}.", invalid_penalty, warnings

        return f"Unsupported action type {action.action_type.value}.", 0.1, [
            "Unsupported action type.",
        ]

    def _require_ticket(self, ticket_id: str | None) -> tuple[TicketRecord | None, str]:
        if not ticket_id:
            return None, "This action requires a ticket_id."
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return None, f"Unknown ticket_id {ticket_id!r}."
        return ticket, ""

    def _build_state(self, episode_id: str, step_count: int) -> SupportOpsState:
        assert self._current_task is not None
        return SupportOpsState(
            episode_id=episode_id,
            step_count=step_count,
            task=self._current_task.task_card(),
            tickets=self._ordered_tickets(),
            current_score=self._current_grade.score,
            criteria=self._current_grade.criteria,
            finished=self._done,
            last_action_result=self._last_action_result,
            last_reward_breakdown=self._last_reward_breakdown,
            action_history=self._action_history.copy(),
        )

    def _build_observation(
        self,
        reward: float,
        warnings: list[str],
    ) -> SupportOpsObservation:
        assert self._current_task is not None
        return SupportOpsObservation(
            task=self._current_task.task_card(),
            inbox=self._ordered_tickets(),
            last_action_result=self._last_action_result,
            remaining_steps=max(0, self._current_task.max_steps - self.state.step_count),
            warnings=warnings,
            reward_breakdown=self._last_reward_breakdown,
            done=self._done,
            reward=reward,
            metadata={
                "score": self._current_grade.score,
                "criteria": [criterion.model_dump(mode="json") for criterion in self._current_grade.criteria],
                "task_id": self._current_task.task_id,
            },
        )

    def _ordered_tickets(self) -> list[TicketRecord]:
        return [
            self._tickets[ticket_id].model_copy(deep=True)
            for ticket_id in sorted(self._tickets)
        ]
