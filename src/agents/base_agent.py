# agents.py
# ---------
"""
Simple agent abstractions for the King-and-Rook vs King environment (or any
turn-based game with the same interface).

Classes
-------
Agent   – abstract base class that queues *plans* of actions.
Random  – concrete agent that picks a random action each turn.
"""

from __future__ import annotations

import random
from typing import Any, List

import numpy as np


class Agent:
    """
    Common scaffolding for any agent that acts in discrete turns.

    The agent maintains four running logs that can be inspected after a
    simulation:

    * ``self.states``   – a record of observed states
    * ``self.actions``  – a record of executed actions
    * ``self.rewards``  – scalar reward after *each* action
    * ``self.dones``    – boolean flag indicating end of episode

    Sub-classes **must** implement :py:meth:`program`, which generates (or
    regenerates) *self.plan* – a list of future actions.  The default
    :py:meth:`make_decision` simply pops the first action in that plan.
    """

    # ------------------------------------------------------------------ #
    #                        Construction / reset                        #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        # Plan (queue) and episode logs
        self.plan: List[Any] = []
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = [np.nan]
        self.dones: List[bool] = [False]

        self.turn: int = 0  # counts the agent's own turns

    # ------------------------------------------------------------------ #
    #                     Core interaction with the world                #
    # ------------------------------------------------------------------ #
    def make_decision(self) -> Any:
        """
        Return the next action to perform.

        If no plan exists, :py:meth:`program` is invoked to replenish it.
        """
        if not self.plan:                       # need a new plan?
            self.program()

        try:
            action = self.plan.pop(0)           # FIFO: take first planned action
        except IndexError as exc:
            state = self.states[-1] if self.states else "unknown"
            raise RuntimeError(
                f"Empty plan – no action available for state {state}"
            ) from exc

        self.turn += 1
        return action

    # ------------------------------------------------------------------ #
    #             Methods to be customised by concrete agents            #
    # ------------------------------------------------------------------ #
    def program(self) -> None:  # noqa: D401
        """Populate *self.plan* with one or more future actions."""
        raise NotImplementedError(
            "Sub-classes must implement the `program()` method."
        )

    # ------------------------------------------------------------------ #
    #                          House-keeping                              #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Alias that clears all internal logs."""
        self.restart()

    def restart(self) -> None:
        """Clear episode logs and start fresh."""
        self.plan.clear()
        self.states.clear()
        self.actions.clear()
        self.rewards = [np.nan]
        self.dones = [False]
        self.turn = 0


# --------------------------------------------------------------------------- #
#                           A random-policy agent                             #
# --------------------------------------------------------------------------- #
class Random(Agent):
    """
    Agent that selects an action uniformly at random from *choices* each turn.
    """

    def __init__(self, choices: List[Any]) -> None:
        super().__init__()
        self.choices = choices

    def program(self) -> None:  # noqa: D401
        """Generate a single random action and store it in *self.plan*."""
        self.plan.append(random.choice(self.choices))
