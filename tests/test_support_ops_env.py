from support_ops_env.models import ActionType, Priority, SupportOpsAction, Team, TicketStatus
from support_ops_env.trainer_env import SupportOpsTrainerEnv


def test_easy_task_can_reach_perfect_score():
    env = SupportOpsTrainerEnv("easy_access_recovery")
    env.reset()

    env.step(
        SupportOpsAction(
            action_type=ActionType.UPDATE_TICKET,
            ticket_id="T-1001",
            priority=Priority.HIGH,
            assigned_team=Team.IDENTITY_SUPPORT,
            status=TicketStatus.WAITING_ON_CUSTOMER,
            tags_to_add=["authentication", "sso"],
        )
    )
    observation, reward, done, info = env.step(
        SupportOpsAction(
            action_type=ActionType.REPLY_TO_CUSTOMER,
            ticket_id="T-1001",
            response_text=(
                "Thanks for flagging this. Because this looks like an SSO assignment issue, "
                "I am routing it to Identity Support. Please reply with your full email "
                "address, the workspace URL, and a screenshot of the error."
            ),
        )
    )
    assert reward > 0
    assert not done
    assert 0.99 < round(info["score"], 4) < 1.0
    assert observation.last_action_result.startswith("Sent customer reply")


def test_medium_task_partial_score_is_dense():
    env = SupportOpsTrainerEnv("medium_refund_recovery")
    env.reset()
    _, reward, done, info = env.step(
        SupportOpsAction(
            action_type=ActionType.UPDATE_TICKET,
            ticket_id="T-2001",
            assigned_team=Team.BILLING_OPS,
            status=TicketStatus.RESOLVED,
            resolution_code="approved_refund",
            tags_to_add=["refund", "downgrade"],
        )
    )
    assert reward > 0
    assert not done
    assert 0.0 < info["score"] < 1.0


def test_medium_task_can_reach_perfect_score():
    env = SupportOpsTrainerEnv("medium_refund_recovery")
    env.reset()
    actions = [
        SupportOpsAction(
            action_type=ActionType.UPDATE_TICKET,
            ticket_id="T-2001",
            priority=Priority.MEDIUM,
            assigned_team=Team.BILLING_OPS,
            status=TicketStatus.RESOLVED,
            resolution_code="approved_refund",
            tags_to_add=["refund", "downgrade"],
        ),
        SupportOpsAction(
            action_type=ActionType.ADD_INTERNAL_NOTE,
            ticket_id="T-2001",
            note_text=(
                "Renewed 3 days ago, still within the 7-day grace window, and the "
                "customer reports no material usage since renewal."
            ),
        ),
        SupportOpsAction(
            action_type=ActionType.REPLY_TO_CUSTOMER,
            ticket_id="T-2001",
            response_text=(
                "We approved the refund today. It will return to your original payment "
                "method in 5 to 10 business days, and we will move the workspace to "
                "Starter immediately."
            ),
        ),
        SupportOpsAction(action_type=ActionType.FINISH),
    ]
    done = False
    for action in actions:
        _, _, done, _ = env.step(action)
    assert done
    assert 0.99 < round(env.state().current_score, 4) < 1.0


def test_hard_task_scripted_rollout_scores_perfectly():
    env = SupportOpsTrainerEnv("hard_export_incident")
    env.reset()
    actions = [
        SupportOpsAction(
            action_type=ActionType.UPDATE_TICKET,
            ticket_id="T-3001",
            priority=Priority.URGENT,
            assigned_team=Team.ENGINEERING,
            status=TicketStatus.WAITING_ON_INTERNAL,
            tags_to_add=["known_incident", "csv_export"],
        ),
        SupportOpsAction(
            action_type=ActionType.ADD_INTERNAL_NOTE,
            ticket_id="T-3001",
            note_text=(
                "Known incident affecting multiple accounts. Acme Health and Northwind Freight "
                "both started failing around 09:10 UTC. Engaged engineering and routed the VIP case."
            ),
        ),
        SupportOpsAction(
            action_type=ActionType.REPLY_TO_CUSTOMER,
            ticket_id="T-3001",
            response_text=(
                "We have an active incident on CSV exports and engineering is already working it. "
                "While we restore normal service, please use Scheduled Reports > Deliveries as the workaround."
            ),
        ),
        SupportOpsAction(
            action_type=ActionType.MERGE_TICKET,
            ticket_id="T-3002",
            target_ticket_id="T-3001",
        ),
        SupportOpsAction(
            action_type=ActionType.UPDATE_TICKET,
            ticket_id="T-3003",
            priority=Priority.HIGH,
            assigned_team=Team.ENGINEERING,
            status=TicketStatus.WAITING_ON_INTERNAL,
            tags_to_add=["known_incident", "csv_export"],
        ),
        SupportOpsAction(
            action_type=ActionType.REPLY_TO_CUSTOMER,
            ticket_id="T-3003",
            response_text=(
                "Thanks for reporting this. We have a known incident affecting CSV exports, "
                "and you can use Scheduled Reports > Deliveries until the service recovers."
            ),
        ),
        SupportOpsAction(action_type=ActionType.FINISH),
    ]
    done = False
    for action in actions:
        _, _, done, _ = env.step(action)
    state = env.state()
    assert done
    assert 0.99 < round(state.current_score, 4) < 1.0
    untouched = next(ticket for ticket in state.tickets if ticket.ticket_id == "T-3004")
    assert untouched.tags == ["billing"]
    assert untouched.replies == []
