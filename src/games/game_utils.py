from __future__ import annotations

from copy import deepcopy
from numpy.random import Generator, default_rng
from typing import Any, Dict, Generic, List, Optional, Protocol, Sequence, Tuple, TypeVar

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agents.base_classes import (
    GameProtocol, PlayerProtocol, EncoderProtocol
)

# --------------------------------------------------------------------------- #
#                                Protocols                                    #
# --------------------------------------------------------------------------- #

S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type

# --------------------------------------------------------------------------- #
#                       Gymnasium-compatible Environment                       #
# --------------------------------------------------------------------------- #

class GymEnvFromGameAndPlayer2(gym.Env, Generic[S, A]):
    """
    A Gymnasium environment adapter around a two-player game and a fixed opponent.

    One `step(action)` applies the *agent's* action first. If the state is not
    terminal, the opponent acts immediately. The returned observation reflects
    the state *after* the opponent move (or after the agent move if terminal).

    Parameters
    ----------
    game : GameProtocol[S, A]
        The underlying game.
    other_player : PlayerProtocol[S, A]
        The opponent that moves after the agent.
    encoder : EncoderProtocol[S, A]
        Provides `observation_space`, `action_space`, and (de)coders.
    max_steps : Optional[int]
        If provided, `truncated=True` when step count reaches this limit.
    rng : Optional[Generator]
        `numpy.random.Generator` used for sampling.  If *None*, the policy
        creates an independent generator via `default_rng()`.
    logger : Optional[logging.Logger]
        Logger for debug/errors.

    Returns (Gym API)
    -----------------
    reset(seed, options) -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        game: GameProtocol[S, A],
        other_player: PlayerProtocol[S, A],
        encoder: EncoderProtocol[S, A],
        *,
        rng: Optional[Generator] = None,
        max_steps: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        self.other_player = other_player
        self._initial_game = deepcopy(game)
        self.game: GameProtocol[S, A] = game
        self.state: S = self.game.initial_state

        # Spaces are provided by the encoder
        self.encoder = encoder
        self.observation_space = encoder.observation_space
        self.action_space = encoder.action_space

        self.max_steps = max_steps
        self._steps = 0
        self.rng: Generator = rng or default_rng()
        self._log = logger or logging.getLogger(__name__)

    # ------------------------------ Gym API -------------------------------- #

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and opponent. Return (observation, info)."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.game = deepcopy(self._initial_game)
        self.state = self.game.initial_state
        self._steps = 0

        # Opponent reset + initial bookkeeping
        self.other_player.reset()
        if not hasattr(self.other_player, "states") or self.other_player.states is None:
            self.other_player.states = []  # type: ignore[attr-defined]
        self.other_player.states.append(self.state)

        obs = self.encoder.encode_obs(self.state)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Apply agent action, then (if needed) the opponent action.
        Returns (obs, reward, terminated, truncated, info).
        """
        # Decode action into domain action if needed
        valid_actions = self.game.actions(self.state)
        domain_action: A = self._decode_action(action, valid_actions)

        # --- Agent move ---
        try:
            new_state = self.game.result(self.state, domain_action)
        except Exception:
            self._log.exception("Error applying agent action %r in state %r", action, self.state)
            raise

        # Reward is for the *player who acted* in this transition
        reward_first = self.game.utility(new_state)
        reward: float = float(reward_first) if reward_first is not None else 0.0
        terminated = self.game.is_terminal(new_state)
        truncated = False
        opponent_action: Optional[A] = None

        # --- Opponent move (if not terminal) ---
        if not terminated:
            self.other_player.states.append(new_state)

            # Update opponent choices if exposed
            if hasattr(self.other_player, "choices"):
                try:
                    self.other_player.choices = self.game.actions(new_state)  # type: ignore[attr-defined]
                except Exception:
                    self._log.debug("Could not update other_player.choices", exc_info=True)

            opponent_action = self.other_player.make_decision()
            if getattr(self.other_player, "debug", False):
                self._log.debug("Opponent plays: %r", opponent_action)

            try:
                new_state = self.game.result(new_state, opponent_action)
            except Exception:
                self._log.exception("Error applying opponent action %r", opponent_action)
                raise

            terminated = self.game.is_terminal(new_state)

        # Bookkeeping
        self.state = new_state
        self._steps += 1

        if self.max_steps is not None and self._steps >= self.max_steps and not terminated:
            truncated = True

        obs = self.encoder.encode_obs(self.state)
        info: Dict[str, Any] = {}
        if opponent_action is not None:
            info["opponent_action"] = self.encoder.encode_action(opponent_action)
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        self.game.render(self.state)

    def close(self) -> None:
        pass

    # ---------------------------- Internals --------------------------------- #

    def _decode_action(self, action: Any, valid_actions: Sequence[A]) -> A:
        """
        Decode an external (e.g., int) action into a domain action using the encoder.
        If the encoder has a `decode_action`, use it; else assume the action is already A.
        """
        if hasattr(self.encoder, "decode_action"):
            # type: ignore[attr-defined]
            return self.encoder.decode_action(self.state, action)  # type: ignore[return-value]
        # Fallback: assume action is already a valid domain action
        return action_any  # type: ignore[return-value]

    @staticmethod
    def _previous_player(next_player: Any) -> Any:
        """
        Helper to infer the last actor given the next player.
        If your game exposes a direct way to get the last actor, replace this.
        """
        # Default: for two-player alternating games with players {0,1}
        if isinstance(next_player, (int, np.integer)) and next_player in (0, 1):
            return 1 - int(next_player)
        # Otherwise, just return a symbolic marker; adjust if your game needs it.
        return "previous"
