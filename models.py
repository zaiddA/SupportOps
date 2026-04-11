# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the Support Ops environment."""

from __future__ import annotations

from enum import Enum

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Supported task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TicketStatus(str, Enum):
    """Ticket workflow stages used in the environment."""

    NEW = "new"
    IN_PROGRESS = "in_progress"
    WAITING_ON_CUSTOMER = "waiting_on_customer"
    WAITING_ON_INTERNAL = "waiting_on_internal"
    RESOLVED = "resolved"
    CLOSED_DUPLICATE = "closed_duplicate"


class Priority(str, Enum):
    """Ticket priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Team(str, Enum):
    """Teams that own support tickets."""

    TRIAGE = "triage"
    IDENTITY_SUPPORT = "identity_support"
    BILLING_OPS = "billing_ops"
    ENGINEERING = "engineering"


class ActionType(str, Enum):
    """Supported agent actions."""

    UPDATE_TICKET = "update_ticket"
    REPLY_TO_CUSTOMER = "reply_to_customer"
    ADD_INTERNAL_NOTE = "add_internal_note"
    MERGE_TICKET = "merge_ticket"
    FINISH = "finish"


class PolicySnippet(BaseModel):
    """Visible policy or knowledge base item for the current task."""

    title: str = Field(..., description="Short policy title")
    content: str = Field(..., description="Policy content visible to the agent")


class TicketRecord(BaseModel):
    """A support ticket as seen by the agent and grader."""

    ticket_id: str = Field(..., description="Stable ticket identifier")
    subject: str = Field(..., description="Ticket subject line")
    customer_name: str = Field(..., description="Customer or requester name")
    company: str = Field(..., description="Company associated with the ticket")
    plan: str = Field(..., description="Customer plan or subscription tier")
    vip: bool = Field(default=False, description="Whether the customer is VIP")
    submitted_at: str = Field(..., description="Submission timestamp")
    summary: str = Field(..., description="Short summary of the issue")
    body: str = Field(..., description="Full customer request body")
    status: TicketStatus = Field(default=TicketStatus.NEW, description="Current ticket status")
    priority: Priority = Field(default=Priority.MEDIUM, description="Current ticket priority")
    assigned_team: Team = Field(default=Team.TRIAGE, description="Current owning team")
    tags: list[str] = Field(default_factory=list, description="Tags applied to the ticket")
    resolution_code: str | None = Field(default=None, description="Resolution code if ticket is resolved")
    merged_into: str | None = Field(default=None, description="Parent ticket id if this ticket was merged")
    replies: list[str] = Field(default_factory=list, description="Customer-facing replies sent so far")
    internal_notes: list[str] = Field(default_factory=list, description="Internal notes added so far")


class TaskCard(BaseModel):
    """High-level task description delivered to the agent."""

    task_id: str = Field(..., description="Stable task identifier")
    title: str = Field(..., description="Task title")
    difficulty: Difficulty = Field(..., description="Difficulty level")
    objective: str = Field(..., description="Concrete task objective")
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Human-readable success criteria for the task",
    )
    knowledge_base: list[PolicySnippet] = Field(
        default_factory=list,
        description="Policy snippets and task instructions available to the agent",
    )
    max_steps: int = Field(..., description="Maximum steps before timeout")


class CriterionScore(BaseModel):
    """Score for one deterministic grading criterion."""

    criterion_id: str = Field(..., description="Stable criterion identifier")
    description: str = Field(..., description="What the criterion checks")
    weight: float = Field(..., description="Criterion weight in the final grade")
    score: float = Field(..., ge=0.0, le=1.0, description="Criterion score from 0.0 to 1.0")
    satisfied: bool = Field(..., description="Whether the criterion is fully satisfied")
    details: str = Field(..., description="Short explanation of the score")


class SupportOpsReward(BaseModel):
    """Typed reward breakdown for one environment step."""

    total: float = Field(..., description="Final scalar reward for the step")
    score_before: float = Field(..., ge=0.0, le=1.0, description="Task score before the action")
    score_after: float = Field(..., ge=0.0, le=1.0, description="Task score after the action")
    score_delta: float = Field(..., description="Difference between score_after and score_before")
    action_cost: float = Field(..., description="Per-action time cost penalty")
    invalid_action_penalty: float = Field(..., description="Penalty for invalid or destructive actions")
    loop_penalty: float = Field(..., description="Penalty for repeating the same action")
    finish_bonus: float = Field(..., description="Bonus or penalty applied when the agent finishes")
    notes: list[str] = Field(default_factory=list, description="Human-readable reward notes")


class ActionRecord(BaseModel):
    """Audit log entry for one step."""

    step_index: int = Field(..., description="1-based step index")
    action_type: ActionType = Field(..., description="Action performed")
    ticket_id: str | None = Field(default=None, description="Primary ticket involved in the action")
    summary: str = Field(..., description="Short summary of what happened")
    reward: SupportOpsReward = Field(..., description="Reward breakdown for this action")
    score_after: float = Field(..., ge=0.0, le=1.0, description="Task score after the step")


class SupportOpsAction(Action):
    """Action emitted by the agent."""

    action_type: ActionType = Field(..., description="Type of support workflow action to execute")
    ticket_id: str | None = Field(default=None, description="Target ticket id for the action")
    status: TicketStatus | None = Field(default=None, description="Updated ticket status")
    priority: Priority | None = Field(default=None, description="Updated ticket priority")
    assigned_team: Team | None = Field(default=None, description="Updated owning team")
    tags_to_add: list[str] = Field(default_factory=list, description="Tags to add to the ticket")
    tags_to_remove: list[str] = Field(default_factory=list, description="Tags to remove from the ticket")
    resolution_code: str | None = Field(default=None, description="Resolution code to set")
    response_text: str | None = Field(default=None, description="Customer-facing reply text")
    note_text: str | None = Field(default=None, description="Internal note text")
    target_ticket_id: str | None = Field(default=None, description="Merge target for duplicate tickets")


class SupportOpsObservation(Observation):
    """Observation returned after reset or step."""

    task: TaskCard = Field(..., description="Current task card and visible policies")
    inbox: list[TicketRecord] = Field(..., description="Current ticket workspace")
    last_action_result: str = Field(..., description="Short description of the last action outcome")
    remaining_steps: int = Field(..., ge=0, description="Steps remaining before timeout")
    warnings: list[str] = Field(default_factory=list, description="Warnings about invalid or risky actions")
    reward_breakdown: SupportOpsReward | None = Field(
        default=None,
        description="Typed reward breakdown for the most recent step",
    )


class SupportOpsState(State):
    """Full internal environment state."""

    task: TaskCard = Field(..., description="Current task definition exposed as a task card")
    tickets: list[TicketRecord] = Field(..., description="Complete ticket workspace state")
    current_score: float = Field(..., gt=0.0, lt=1.0, description="Current deterministic task score")
    criteria: list[CriterionScore] = Field(..., description="Criterion-by-criterion grading details")
    finished: bool = Field(..., description="Whether the episode has finished")
    last_action_result: str = Field(..., description="Most recent action outcome message")
    last_reward_breakdown: SupportOpsReward | None = Field(
        default=None,
        description="Reward breakdown from the last executed step",
    )
    action_history: list[ActionRecord] = Field(
        default_factory=list,
        description="Ordered audit log of actions taken in the episode",
    )
