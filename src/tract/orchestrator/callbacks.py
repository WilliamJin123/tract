"""Built-in proposal review callbacks for the orchestrator.

Provides ready-made callbacks for common approval workflows:
auto_approve, log_and_approve, cli_prompt, and reject_all.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from tract.orchestrator.models import (
    OrchestratorProposal,
    ProposalDecision,
    ProposalResponse,
    ToolCall,
)

logger = logging.getLogger(__name__)


def auto_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Approve any proposal automatically.

    For autonomous mode where no human review is needed.
    """
    return ProposalResponse(decision=ProposalDecision.APPROVED)


def log_and_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Log proposal details then approve automatically.

    For audit trail mode -- all actions are approved but logged
    for later review.
    """
    logger.info(
        "Orchestrator proposal %s: action=%s, reasoning=%s",
        proposal.proposal_id,
        proposal.recommended_action.name,
        proposal.reasoning,
    )
    return ProposalResponse(decision=ProposalDecision.APPROVED)


def cli_prompt(proposal: OrchestratorProposal) -> ProposalResponse:
    """Interactive CLI prompt for proposal review.

    Uses Rich for display if available (optional [cli] extra),
    falls back to plain input() otherwise.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        content = (
            f"[bold]Action:[/bold] {proposal.recommended_action.name}\n"
            f"[bold]Reasoning:[/bold] {proposal.reasoning}\n"
            f"[bold]Arguments:[/bold] {json.dumps(proposal.recommended_action.arguments, indent=2)}"
        )
        console.print(Panel(content, title=f"Proposal: {proposal.proposal_id}"))
        prompt_fn = console.input
    except ImportError:
        print(f"\n--- Proposal: {proposal.proposal_id} ---")
        print(f"Action: {proposal.recommended_action.name}")
        print(f"Reasoning: {proposal.reasoning}")
        print(f"Arguments: {json.dumps(proposal.recommended_action.arguments, indent=2)}")
        prompt_fn = input

    try:
        while True:
            choice = prompt_fn("[a]pprove / [r]eject / [m]odify: ").strip().lower()

            if choice in ("a", "approve"):
                return ProposalResponse(decision=ProposalDecision.APPROVED)

            if choice in ("r", "reject"):
                reason = prompt_fn("Reason (optional): ").strip()
                return ProposalResponse(
                    decision=ProposalDecision.REJECTED, reason=reason
                )

            if choice in ("m", "modify"):
                while True:
                    raw = prompt_fn("Modified arguments (JSON): ").strip()
                    try:
                        parsed_args = json.loads(raw)
                    except json.JSONDecodeError:
                        print("Invalid JSON. Please try again.")
                        continue
                    return ProposalResponse(
                        decision=ProposalDecision.MODIFIED,
                        modified_action=ToolCall(
                            id=proposal.recommended_action.id,
                            name=proposal.recommended_action.name,
                            arguments=parsed_args,
                        ),
                    )

            print("Invalid choice. Enter 'a', 'r', or 'm'.")
    except (EOFError, KeyboardInterrupt):
        return ProposalResponse(
            decision=ProposalDecision.REJECTED, reason="Input closed"
        )


def reject_all(proposal: OrchestratorProposal) -> ProposalResponse:
    """Reject any proposal automatically.

    For testing and safety -- blocks all orchestrator actions.
    """
    return ProposalResponse(
        decision=ProposalDecision.REJECTED, reason="Auto-rejected"
    )
