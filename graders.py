# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic graders for the Support Ops environment."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .models import CriterionScore, TicketRecord
from .tasks import CriterionKind, TaskCriterion, TaskDefinition, TextSource


class TaskGrade(BaseModel):
    """Current score and criterion breakdown for a task."""

    score: float = Field(..., ge=0.0, le=1.0, description="Current task score")
    criteria: list[CriterionScore] = Field(default_factory=list, description="Criterion breakdown")


def grade_task(
    task: TaskDefinition,
    tickets: dict[str, TicketRecord],
    initial_tickets: dict[str, TicketRecord],
) -> TaskGrade:
    """Grade the current task state deterministically."""

    weighted_total = 0.0
    weight_sum = 0.0
    criteria_scores: list[CriterionScore] = []

    for criterion in task.criteria:
        score = _score_criterion(criterion, tickets, initial_tickets)
        weighted_total += score.weight * score.score
        weight_sum += score.weight
        criteria_scores.append(score)

    overall = 0.0 if weight_sum == 0.0 else weighted_total / weight_sum
    return TaskGrade(score=round(_clamp(overall), 4), criteria=criteria_scores)


def _score_criterion(
    criterion: TaskCriterion,
    tickets: dict[str, TicketRecord],
    initial_tickets: dict[str, TicketRecord],
) -> CriterionScore:
    ticket = tickets.get(criterion.ticket_id)
    if ticket is None:
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=0.0,
            satisfied=False,
            details=f"Ticket {criterion.ticket_id} not found.",
        )

    if criterion.kind == CriterionKind.FIELD:
        actual = getattr(ticket, criterion.field_name or "", None)
        actual_value = str(actual.value if hasattr(actual, "value") else actual)
        expected = criterion.expected_value
        if isinstance(expected, list):
            satisfied = actual_value in expected
            details = f"Expected one of {expected}; found {actual_value!r}."
        else:
            expected_value = str(expected)
            satisfied = actual_value == expected_value
            details = f"Expected {expected_value!r}; found {actual_value!r}."
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=1.0 if satisfied else 0.0,
            satisfied=satisfied,
            details=details,
        )

    if criterion.kind == CriterionKind.TAGS:
        current_tags = set(ticket.tags)
        required_hits = sum(tag in current_tags for tag in criterion.required_tags)
        required_score = (
            1.0 if not criterion.required_tags else required_hits / len(criterion.required_tags)
        )
        forbidden_hits = [tag for tag in criterion.forbidden_tags if tag in current_tags]
        penalty = min(0.5, 0.25 * len(forbidden_hits)) if forbidden_hits else 0.0
        score = _clamp(required_score - penalty)
        details = (
            f"Tags present: {sorted(current_tags)}. "
            f"Required matched: {required_hits}/{max(1, len(criterion.required_tags))}."
        )
        if forbidden_hits:
            details += f" Forbidden tags present: {forbidden_hits}."
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=round(score, 4),
            satisfied=score >= 0.999,
            details=details,
        )

    if criterion.kind == CriterionKind.TEXT:
        source = criterion.source or TextSource.REPLY
        parts = ticket.replies if source == TextSource.REPLY else ticket.internal_notes
        normalized_text = _normalize_text(" ".join(parts))
        if not normalized_text:
            return CriterionScore(
                criterion_id=criterion.criterion_id,
                description=criterion.description,
                weight=criterion.weight,
                score=0.0,
                satisfied=False,
                details=f"No {source.value.lower()} text found.",
            )
        matched_groups = 0
        for group in criterion.keyword_groups:
            normalized_options = [_normalize_text(item) for item in group]
            if any(option and option in normalized_text for option in normalized_options):
                matched_groups += 1
        base_score = (
            1.0
            if not criterion.keyword_groups
            else matched_groups / len(criterion.keyword_groups)
        )
        forbidden_hits = []
        for phrase in criterion.forbidden_keywords:
            normalized_phrase = _normalize_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_text:
                forbidden_hits.append(phrase)
        penalty = min(0.6, 0.25 * len(forbidden_hits))
        score = _clamp(base_score - penalty)
        details = (
            f"Matched {matched_groups}/{max(1, len(criterion.keyword_groups))} required concepts."
        )
        if forbidden_hits:
            details += f" Forbidden phrases present: {forbidden_hits}."
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=round(score, 4),
            satisfied=score >= 0.999,
            details=details,
        )

    if criterion.kind == CriterionKind.MERGE:
        expected_value = str(criterion.expected_value)
        satisfied = ticket.merged_into == expected_value
        details = f"Expected merge target {expected_value!r}; found {ticket.merged_into!r}."
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=1.0 if satisfied else 0.0,
            satisfied=satisfied,
            details=details,
        )

    if criterion.kind == CriterionKind.UNTOUCHED:
        initial = initial_tickets.get(criterion.ticket_id)
        if initial is None:
            return CriterionScore(
                criterion_id=criterion.criterion_id,
                description=criterion.description,
                weight=criterion.weight,
                score=0.0,
                satisfied=False,
                details=f"Initial ticket {criterion.ticket_id} not found.",
            )
        mutable_snapshot = _mutable_ticket_snapshot(ticket)
        initial_snapshot = _mutable_ticket_snapshot(initial)
        satisfied = mutable_snapshot == initial_snapshot
        details = "Ticket remained untouched." if satisfied else "Ticket state changed from the initial snapshot."
        return CriterionScore(
            criterion_id=criterion.criterion_id,
            description=criterion.description,
            weight=criterion.weight,
            score=1.0 if satisfied else 0.0,
            satisfied=satisfied,
            details=details,
        )

    raise ValueError(f"Unsupported criterion kind: {criterion.kind}")


def _mutable_ticket_snapshot(ticket: TicketRecord) -> tuple[object, ...]:
    return (
        ticket.status.value,
        ticket.priority.value,
        ticket.assigned_team.value,
        tuple(sorted(ticket.tags)),
        ticket.resolution_code,
        ticket.merged_into,
        tuple(ticket.replies),
        tuple(ticket.internal_notes),
    )


def _normalize_text(value: str) -> str:
    alphanumeric_only = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", alphanumeric_only).strip()


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
