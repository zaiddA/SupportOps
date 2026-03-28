# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task definitions for the Support Ops environment."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .models import (
    Difficulty,
    PolicySnippet,
    Priority,
    TaskCard,
    Team,
    TicketRecord,
    TicketStatus,
)


class CriterionKind(str, Enum):
    """Types of grader criteria."""

    FIELD = "field"
    TAGS = "tags"
    TEXT = "text"
    MERGE = "merge"
    UNTOUCHED = "untouched"


class TextSource(str, Enum):
    """Where a text criterion should look."""

    REPLY = "reply"
    NOTE = "note"


class TaskCriterion(BaseModel):
    """One deterministic grading rule."""

    criterion_id: str = Field(..., description="Stable criterion identifier")
    description: str = Field(..., description="Human-readable criterion description")
    weight: float = Field(..., gt=0.0, description="Weight in final task score")
    kind: CriterionKind = Field(..., description="Criterion type")
    ticket_id: str = Field(..., description="Ticket the criterion applies to")
    field_name: str | None = Field(default=None, description="Field name for field criteria")
    expected_value: str | list[str] | None = Field(
        default=None,
        description="Expected value for field or merge criteria",
    )
    required_tags: list[str] = Field(default_factory=list, description="Tags that must be present")
    forbidden_tags: list[str] = Field(default_factory=list, description="Tags that must not be present")
    source: TextSource | None = Field(default=None, description="Source for text criteria")
    keyword_groups: list[list[str]] = Field(
        default_factory=list,
        description="Each inner list contains acceptable phrases for one required concept",
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list,
        description="Phrases that should not appear in text criteria",
    )


class TaskDefinition(BaseModel):
    """A complete environment task with initial workspace state and grader."""

    task_id: str = Field(..., description="Stable task identifier")
    title: str = Field(..., description="Task title")
    difficulty: Difficulty = Field(..., description="Difficulty level")
    objective: str = Field(..., description="Task objective")
    success_criteria: list[str] = Field(default_factory=list, description="Visible success criteria")
    knowledge_base: list[PolicySnippet] = Field(default_factory=list, description="Task policy snippets")
    max_steps: int = Field(..., gt=0, description="Maximum steps for the task")
    tickets: list[TicketRecord] = Field(..., description="Initial ticket workspace")
    criteria: list[TaskCriterion] = Field(..., description="Deterministic grader criteria")

    def task_card(self) -> TaskCard:
        """Return the visible task card for the agent."""

        return TaskCard(
            task_id=self.task_id,
            title=self.title,
            difficulty=self.difficulty,
            objective=self.objective,
            success_criteria=self.success_criteria,
            knowledge_base=self.knowledge_base,
            max_steps=self.max_steps,
        )


EASY_TASK_ID = "easy_access_recovery"
MEDIUM_TASK_ID = "medium_refund_recovery"
HARD_TASK_ID = "hard_export_incident"
TASK_ORDER = [EASY_TASK_ID, MEDIUM_TASK_ID, HARD_TASK_ID]


def build_task_catalog() -> dict[str, TaskDefinition]:
    """Build the deterministic task catalog."""

    return {
        EASY_TASK_ID: _build_easy_task(),
        MEDIUM_TASK_ID: _build_medium_task(),
        HARD_TASK_ID: _build_hard_task(),
    }


def _build_easy_task() -> TaskDefinition:
    return TaskDefinition(
        task_id=EASY_TASK_ID,
        title="Restore access for a blocked SSO user",
        difficulty=Difficulty.EASY,
        objective=(
            "A new employee at Acme Health is blocked by an SSO configuration issue. "
            "Triage the ticket to the correct team, set the right urgency, and send a "
            "customer reply that gathers the exact details Identity Support needs."
        ),
        success_criteria=[
            "Route the ticket to Identity Support with high priority.",
            "Leave the ticket waiting on the customer instead of resolving it.",
            "Ask for the workspace URL, the user's full email address, and a screenshot.",
        ],
        knowledge_base=[
            PolicySnippet(
                title="SSO routing",
                content="Account access problems tied to Okta or SSO belong to Identity Support.",
            ),
            PolicySnippet(
                title="Access urgency",
                content="Users blocked from signing in during onboarding should be handled as high priority.",
            ),
            PolicySnippet(
                title="Required details",
                content="For SSO assignment issues, ask for the workspace URL, the full email address, and a screenshot of the exact error.",
            ),
            PolicySnippet(
                title="Avoid bad guidance",
                content="Do not tell SSO customers to reset a local Nimbus password when the issue is with IdP assignment.",
            ),
        ],
        max_steps=6,
        tickets=[
            TicketRecord(
                ticket_id="T-1001",
                subject="Can't log into Nimbus after SSO switch",
                customer_name="Maya Patel",
                company="Acme Health",
                plan="Business",
                vip=False,
                submitted_at="2026-03-28T09:14:00Z",
                summary="New hire blocked by Okta SSO assignment error.",
                body=(
                    "Hi support, I joined our RevOps team today and can't sign into Nimbus. "
                    "We moved to Okta SSO yesterday. I enter my email and get an "
                    "'app_not_assigned' error. I'm blocked from onboarding today."
                ),
                status=TicketStatus.NEW,
                priority=Priority.MEDIUM,
                assigned_team=Team.TRIAGE,
                tags=["login"],
            )
        ],
        criteria=[
            TaskCriterion(
                criterion_id="easy_priority",
                description="Ticket priority is raised to high",
                weight=0.15,
                kind=CriterionKind.FIELD,
                ticket_id="T-1001",
                field_name="priority",
                expected_value=Priority.HIGH.value,
            ),
            TaskCriterion(
                criterion_id="easy_team",
                description="Ticket is routed to Identity Support",
                weight=0.15,
                kind=CriterionKind.FIELD,
                ticket_id="T-1001",
                field_name="assigned_team",
                expected_value=Team.IDENTITY_SUPPORT.value,
            ),
            TaskCriterion(
                criterion_id="easy_status",
                description="Ticket remains waiting on the customer",
                weight=0.15,
                kind=CriterionKind.FIELD,
                ticket_id="T-1001",
                field_name="status",
                expected_value=TicketStatus.WAITING_ON_CUSTOMER.value,
            ),
            TaskCriterion(
                criterion_id="easy_tags",
                description="Ticket has the expected SSO tags",
                weight=0.15,
                kind=CriterionKind.TAGS,
                ticket_id="T-1001",
                required_tags=["authentication", "sso"],
            ),
            TaskCriterion(
                criterion_id="easy_reply",
                description="Customer reply asks for the right diagnostic details",
                weight=0.40,
                kind=CriterionKind.TEXT,
                ticket_id="T-1001",
                source=TextSource.REPLY,
                keyword_groups=[
                    ["sso", "single sign on", "single sign-on"],
                    ["workspace url", "workspace link"],
                    ["full email", "email address"],
                    ["screenshot"],
                ],
                forbidden_keywords=["reset your password", "password reset"],
            ),
        ],
    )


def _build_medium_task() -> TaskDefinition:
    return TaskDefinition(
        task_id=MEDIUM_TASK_ID,
        title="Approve a renewal refund and downgrade request",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "A customer was accidentally renewed onto an annual plan three days ago and "
            "wants both a refund and a downgrade. Apply the billing policy correctly, "
            "document the rationale internally, and resolve the case."
        ),
        success_criteria=[
            "Route the case to Billing Ops and keep the priority appropriate for a billing workflow.",
            "Document why the refund is eligible under the renewal grace window.",
            "Resolve the ticket with an approved refund and explain what happens next.",
        ],
        knowledge_base=[
            PolicySnippet(
                title="Renewal refund policy",
                content="Annual renewal refunds are allowed within 7 days when the account has had no material usage since renewal.",
            ),
            PolicySnippet(
                title="Queue ownership",
                content="Refund and plan-change cases are owned by Billing Ops.",
            ),
            PolicySnippet(
                title="Payout timing",
                content="Approved card refunds return to the original payment method in 5-10 business days.",
            ),
            PolicySnippet(
                title="Downgrade handling",
                content="When an annual renewal refund is approved, the workspace can be moved to Starter immediately.",
            ),
        ],
        max_steps=7,
        tickets=[
            TicketRecord(
                ticket_id="T-2001",
                subject="Please refund accidental annual renewal",
                customer_name="Jordan Lee",
                company="Northwind Freight",
                plan="Pro Annual",
                vip=False,
                submitted_at="2026-03-28T10:02:00Z",
                summary="Customer wants refund for a recent annual renewal and a downgrade to Starter.",
                body=(
                    "We were charged for another year three days ago and our team hasn't "
                    "used Nimbus since the renewal. Can you refund this and move us down "
                    "to Starter? We only need one seat now."
                ),
                status=TicketStatus.NEW,
                priority=Priority.MEDIUM,
                assigned_team=Team.TRIAGE,
                tags=["billing"],
            )
        ],
        criteria=[
            TaskCriterion(
                criterion_id="medium_priority",
                description="Ticket stays at medium priority",
                weight=0.10,
                kind=CriterionKind.FIELD,
                ticket_id="T-2001",
                field_name="priority",
                expected_value=Priority.MEDIUM.value,
            ),
            TaskCriterion(
                criterion_id="medium_team",
                description="Ticket is routed to Billing Ops",
                weight=0.10,
                kind=CriterionKind.FIELD,
                ticket_id="T-2001",
                field_name="assigned_team",
                expected_value=Team.BILLING_OPS.value,
            ),
            TaskCriterion(
                criterion_id="medium_status",
                description="Ticket is resolved once the refund is approved",
                weight=0.10,
                kind=CriterionKind.FIELD,
                ticket_id="T-2001",
                field_name="status",
                expected_value=TicketStatus.RESOLVED.value,
            ),
            TaskCriterion(
                criterion_id="medium_resolution",
                description="Resolution code marks an approved refund",
                weight=0.10,
                kind=CriterionKind.FIELD,
                ticket_id="T-2001",
                field_name="resolution_code",
                expected_value="approved_refund",
            ),
            TaskCriterion(
                criterion_id="medium_tags",
                description="Refund and downgrade tags are applied",
                weight=0.10,
                kind=CriterionKind.TAGS,
                ticket_id="T-2001",
                required_tags=["billing", "refund", "downgrade"],
            ),
            TaskCriterion(
                criterion_id="medium_note",
                description="Internal note cites why the refund is eligible",
                weight=0.20,
                kind=CriterionKind.TEXT,
                ticket_id="T-2001",
                source=TextSource.NOTE,
                keyword_groups=[
                    ["3 days", "three days"],
                    ["7 day", "seven day", "grace window"],
                    ["no material usage", "no usage", "unused since renewal"],
                ],
            ),
            TaskCriterion(
                criterion_id="medium_reply",
                description="Customer reply explains refund timing and downgrade outcome",
                weight=0.30,
                kind=CriterionKind.TEXT,
                ticket_id="T-2001",
                source=TextSource.REPLY,
                keyword_groups=[
                    ["refund", "reversed"],
                    ["5 10 business days", "5 to 10 business days", "five to ten business days"],
                    ["starter"],
                    ["immediately", "today"],
                ],
            ),
        ],
    )


def _build_hard_task() -> TaskDefinition:
    return TaskDefinition(
        task_id=HARD_TASK_ID,
        title="Coordinate a live CSV export incident",
        difficulty=Difficulty.HARD,
        objective=(
            "Multiple customers are reporting the same CSV export failure. Distinguish the "
            "true duplicate from the separate cross-account incident, route impacted tickets "
            "correctly, communicate the workaround, and leave unrelated work untouched."
        ),
        success_criteria=[
            "Escalate the VIP enterprise ticket to Engineering with urgent priority.",
            "Merge only the same-company duplicate into the most detailed parent ticket.",
            "Mark the other affected account as part of the known incident and share the workaround.",
            "Do not modify the unrelated billing ticket.",
        ],
        knowledge_base=[
            PolicySnippet(
                title="Incident detection",
                content="When the same failure is reported by multiple customer accounts, treat it as a known incident.",
            ),
            PolicySnippet(
                title="Incident tagging",
                content="Use the tags known_incident and csv_export on incident-related tickets.",
            ),
            PolicySnippet(
                title="VIP handling",
                content="VIP customers impacted by an active incident should be urgent and owned by Engineering.",
            ),
            PolicySnippet(
                title="Duplicate workflow",
                content="Only merge duplicate tickets when they come from the same company and refer to the same issue.",
            ),
            PolicySnippet(
                title="Workaround",
                content="During CSV export degradation, suggest Scheduled Reports > Deliveries as the temporary workaround.",
            ),
            PolicySnippet(
                title="Scope discipline",
                content="Unrelated tickets should remain untouched while you handle the incident.",
            ),
        ],
        max_steps=10,
        tickets=[
            TicketRecord(
                ticket_id="T-3001",
                subject="Urgent: export jobs failing with 500 error",
                customer_name="Lena Torres",
                company="Acme Health",
                plan="Enterprise",
                vip=True,
                submitted_at="2026-03-28T09:21:00Z",
                summary="VIP customer cannot export account activity to CSV before a board meeting.",
                body=(
                    "Our finance team can't export account activity to CSV since 09:10 UTC. "
                    "Every run fails with HTTP 500. We need the file before the board meeting today."
                ),
                status=TicketStatus.NEW,
                priority=Priority.HIGH,
                assigned_team=Team.TRIAGE,
                tags=["export", "vip"],
            ),
            TicketRecord(
                ticket_id="T-3002",
                subject="Blank screen after clicking Export",
                customer_name="Omar Ruiz",
                company="Acme Health",
                plan="Enterprise",
                vip=False,
                submitted_at="2026-03-28T09:33:00Z",
                summary="Same company as Lena, likely duplicate export failure.",
                body=(
                    "Same company as Lena. Export is broken for me too and the page goes blank "
                    "after I click the CSV export button."
                ),
                status=TicketStatus.NEW,
                priority=Priority.MEDIUM,
                assigned_team=Team.TRIAGE,
                tags=["export"],
            ),
            TicketRecord(
                ticket_id="T-3003",
                subject="CSV exports timing out",
                customer_name="Priya Nair",
                company="Northwind Freight",
                plan="Business",
                vip=False,
                submitted_at="2026-03-28T09:47:00Z",
                summary="Separate customer reports export failure the same morning.",
                body=(
                    "CSV exports have been failing since this morning. Can someone confirm "
                    "if this is on your side?"
                ),
                status=TicketStatus.NEW,
                priority=Priority.MEDIUM,
                assigned_team=Team.TRIAGE,
                tags=["export"],
            ),
            TicketRecord(
                ticket_id="T-3004",
                subject="Need invoice with PO number",
                customer_name="Evan Ross",
                company="Silverline Labs",
                plan="Starter",
                vip=False,
                submitted_at="2026-03-28T10:15:00Z",
                summary="Unrelated billing request for invoice metadata.",
                body="Can you add PO-4421 to this month's invoice PDF?",
                status=TicketStatus.NEW,
                priority=Priority.MEDIUM,
                assigned_team=Team.TRIAGE,
                tags=["billing"],
            ),
        ],
        criteria=[
            TaskCriterion(
                criterion_id="hard_t1_priority",
                description="VIP incident ticket is marked urgent",
                weight=0.08,
                kind=CriterionKind.FIELD,
                ticket_id="T-3001",
                field_name="priority",
                expected_value=Priority.URGENT.value,
            ),
            TaskCriterion(
                criterion_id="hard_t1_team",
                description="VIP incident ticket is owned by Engineering",
                weight=0.08,
                kind=CriterionKind.FIELD,
                ticket_id="T-3001",
                field_name="assigned_team",
                expected_value=Team.ENGINEERING.value,
            ),
            TaskCriterion(
                criterion_id="hard_t1_status",
                description="VIP incident ticket waits on internal resolution",
                weight=0.07,
                kind=CriterionKind.FIELD,
                ticket_id="T-3001",
                field_name="status",
                expected_value=TicketStatus.WAITING_ON_INTERNAL.value,
            ),
            TaskCriterion(
                criterion_id="hard_t1_tags",
                description="VIP incident ticket has the incident tags",
                weight=0.08,
                kind=CriterionKind.TAGS,
                ticket_id="T-3001",
                required_tags=["known_incident", "csv_export", "vip"],
            ),
            TaskCriterion(
                criterion_id="hard_t1_note",
                description="Internal note captures blast radius and escalation",
                weight=0.10,
                kind=CriterionKind.TEXT,
                ticket_id="T-3001",
                source=TextSource.NOTE,
                keyword_groups=[
                    ["multiple accounts", "more than one account"],
                    ["acme health"],
                    ["northwind freight"],
                    ["09 10 utc", "09:10 utc"],
                    ["engaged engineering", "routing to engineering", "routed to engineering"],
                ],
            ),
            TaskCriterion(
                criterion_id="hard_t1_reply",
                description="VIP reply acknowledges incident, routing, and workaround",
                weight=0.11,
                kind=CriterionKind.TEXT,
                ticket_id="T-3001",
                source=TextSource.REPLY,
                keyword_groups=[
                    ["known incident", "active incident"],
                    ["engineering"],
                    ["scheduled reports", "deliveries"],
                ],
            ),
            TaskCriterion(
                criterion_id="hard_t2_merge",
                description="Same-company duplicate is merged into the parent ticket",
                weight=0.08,
                kind=CriterionKind.MERGE,
                ticket_id="T-3002",
                expected_value="T-3001",
            ),
            TaskCriterion(
                criterion_id="hard_t2_status",
                description="Merged duplicate is closed as duplicate",
                weight=0.06,
                kind=CriterionKind.FIELD,
                ticket_id="T-3002",
                field_name="status",
                expected_value=TicketStatus.CLOSED_DUPLICATE.value,
            ),
            TaskCriterion(
                criterion_id="hard_t3_priority",
                description="Cross-account incident ticket is high priority",
                weight=0.05,
                kind=CriterionKind.FIELD,
                ticket_id="T-3003",
                field_name="priority",
                expected_value=Priority.HIGH.value,
            ),
            TaskCriterion(
                criterion_id="hard_t3_team",
                description="Cross-account incident ticket is routed to Engineering",
                weight=0.05,
                kind=CriterionKind.FIELD,
                ticket_id="T-3003",
                field_name="assigned_team",
                expected_value=Team.ENGINEERING.value,
            ),
            TaskCriterion(
                criterion_id="hard_t3_status",
                description="Cross-account incident ticket waits on internal resolution",
                weight=0.05,
                kind=CriterionKind.FIELD,
                ticket_id="T-3003",
                field_name="status",
                expected_value=TicketStatus.WAITING_ON_INTERNAL.value,
            ),
            TaskCriterion(
                criterion_id="hard_t3_tags",
                description="Cross-account incident ticket has the incident tags",
                weight=0.05,
                kind=CriterionKind.TAGS,
                ticket_id="T-3003",
                required_tags=["known_incident", "csv_export"],
            ),
            TaskCriterion(
                criterion_id="hard_t3_reply",
                description="Cross-account reply shares incident acknowledgement and workaround",
                weight=0.07,
                kind=CriterionKind.TEXT,
                ticket_id="T-3003",
                source=TextSource.REPLY,
                keyword_groups=[
                    ["known incident", "active incident"],
                    ["scheduled reports", "deliveries"],
                ],
            ),
            TaskCriterion(
                criterion_id="hard_t4_untouched",
                description="Unrelated billing ticket remains untouched",
                weight=0.07,
                kind=CriterionKind.UNTOUCHED,
                ticket_id="T-3004",
            ),
        ],
    )
