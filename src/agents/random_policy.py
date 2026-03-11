from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Optional

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng

from agents.base_classes import PolicyProtocol, GameProtocol


class GameUniformPolicy(PolicyProtocol):
    """Policy that returns a uniform distribution over legal actions
    and can be *called* to sample one.

    Parameters
    ----------
    game
        An object implementing `actions(state) -> Sequence[Any]`.
    rng
        `numpy.random.Generator` used for sampling.  If *None*, the policy
        creates an independent generator via `default_rng()`.
    encoder : EncoderProtocol, optional
        An optional state and action encoder.
    """

    __slots__ = ("_game",)

    def __init__(
        self, 
        game: GameProtocol, 
        encoder: Optional[Any] = None,
        rng: Optional[Generator] = None
    ) -> None:
        if not isinstance(game, GameProtocol):
            raise TypeError(
                "`game` must implement the `actions(state) -> Sequence` interface."
            )
        self._game: GameProtocol = game
        self._rng: Generator = rng or default_rng()
        if encoder is not None:
            assert hasattr(encoder, "n_actions"), "`encoder` must have an `n_actions` attribute."
        self.encoder = encoder

    # ------------------------------ CALL ------------------------------------
    def __call__(self, state: Any) -> Any:
        """Sample and return a legal action for *state* using the RNG provided."""
        actions = self._game.actions(state)
        if not actions:
            raise ValueError("No available actions for the given state.")

        probs = self.predict(state)
        return self._rng.choice(actions, p=probs)
    
    # ---------------------------------------------------------------- API
    def predict(self, state: Any) -> NDArray[np.floating]:
        """Return a uniform probability vector over actions available in *state*."""
        
        if self.encoder is not None:
            n_actions = self.encoder.n_actions
        else:
            actions = self._game.actions(state)
            n_actions: int = len(actions)

        if n_actions == 0:
            raise ValueError("No available actions for the given state.")

        return np.full(n_actions, 1.0 / n_actions, dtype=float)

    def predict_in_list(
        self, state: Any, subset: Sequence[Any], renormalize: bool = True
    ) -> NDArray[np.floating]:
        """Return probabilities restricted to *subset* of actions.

        Parameters
        ----------
        state
            Environment state.
        subset
            Iterable of actions to keep.
        renormalize
            If *True* (default), returned probabilities sum to 1; otherwise they
            retain their original mass and may sum to < 1.
        """
        full_probs = self.predict(state)
        full_actions = self._game.actions(state)

        mask = np.array([a in subset for a in full_actions], dtype=bool)
        if not mask.any():
            raise ValueError("`subset` contains no valid actions for this state.")

        sub_probs = full_probs[mask]
        if renormalize:
            sub_probs = sub_probs / sub_probs.sum()  # type: ignore[assignment]

        return sub_probs

    # ---------------------------------------------------------------- No-ops
    def learn(self, *args: Any, **kwargs: Any) -> None:
        """No-op; the policy is fixed and does not learn."""
        return None

    def save(self, file: str | Path) -> None:  # noqa: D401
        """No-op; nothing to save."""
        return None

    def load(self, file: str | Path) -> None:  # noqa: D401
        """No-op; nothing to load."""
        return None
