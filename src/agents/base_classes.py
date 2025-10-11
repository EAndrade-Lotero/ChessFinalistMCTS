from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    List, Optional, Protocol, Sequence, Any, 
    runtime_checkable, Generic, TypeVar
)

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


# --------------------------------------------------------------------------- #
#                                Protocols                                    #
# --------------------------------------------------------------------------- #

S = TypeVar("S")  # State type
A = TypeVar("A")  # Action type



@runtime_checkable
class PolicyProtocol(Protocol):
    """Minimal interface every policy object must implement."""

    def predict(self, state: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def learn(self, *args, **kwargs) -> None: ...
    def save(self, file: str) -> None: ...
    def load(self, file: str) -> None: ...


@runtime_checkable
class GameProtocol(Protocol):
    """Minimal game interface."""

    # Attributes
    initial_state: S

    # Methods
    def is_terminal(self, state: S) -> bool: ...
    def player(self, state: S) -> Any: ...
    def utility(self, state: S) -> Optional[float]: ...
    def actions(self, state: S) -> Sequence[A]: ...
    def result(self, state: S, action: A) -> S: ...
    def render(self, state: S) -> None: ...


@runtime_checkable
class PlayerProtocol(Protocol, Generic[S, A]):
    """Minimal second-player interface expected by GymEnvFromGameAndPlayer2."""

    states: List[S]
    debug: bool
    choices: Sequence[A]

    def reset(self) -> None: ...
    def make_decision(self) -> A: ...

@runtime_checkable
class EncoderProtocol(Protocol, Generic[S, A]):
    """Encodes domain states/actions to gym spaces and back (if needed)."""

    observation_space: spaces.Space
    action_space: spaces.Space

    def encode_obs(self, state: S) -> Any: ...
    # If action is already primitive (int), this can be identity:
    def encode_action(self, action: A) -> Any: ...
    # If you need to map int→domain action:
    def decode_action(self, action: Any, valid_actions: Sequence[A]) -> A: ...


class PolicyAgent(ABC):
    """Base class for agents that act through a *policy* object."""

    def __init__(
        self,
        *,
        policy: PolicyProtocol,
        action_encoder: int,
        rng: Optional[Generator] = None,
        debug: bool = False,
    ) -> None:
        # ---- validate inputs -------------------------------------------------
        if n_actions <= 0:
            raise ValueError("`n_actions` must be a positive integer.")

        self.n_actions: int = n_actions
        self.debug: bool = debug

        # ---- policy ----------------------------------------------------------
        self._policy: PolicyProtocol = deepcopy(policy)
        self._policy_backup: PolicyProtocol = deepcopy(policy)

        # ---- trajectory buffers ---------------------------------------------
        self.states: List[NDArray[np.floating]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = [np.nan]
        self.dones: List[bool] = [False]

        # ---- RNG -------------------------------------------------------------
        self.rng: Generator = rng if rng is not None else default_rng()

    # --------------------------------------------------------------------- API
    def make_decision(self, state: Optional[NDArray[np.floating]] = None) -> int:
        """Return an action sampled from the current policy."""
        if state is None:
            if not self.states:
                raise ValueError("State history is empty; provide `state` explicitly.")
            state = self.states[-1]

        probs = self._policy.predict(state)

        if self.debug:
            if probs.ndim != 1 or probs.size != self.n_actions:
                raise ValueError("`predict` must return a 1-D array of length `n_actions`.")
            if not np.isclose(probs.sum(), 1.0):
                raise ValueError("Action probabilities must sum to 1.")

        return int(self.rng.choice(self.n_actions, p=probs))

    @abstractmethod
    def update(
        self,
        next_state: NDArray[np.floating],
        reward: float,
        done: bool,
    ) -> None:
        """Update policy parameters (must be implemented by subclasses)."""
        ...

    # ---------------------------------------------------------------- helpers
    def restart(self) -> None:
        """Clear trajectory buffers for a **new trial**."""
        self.states.clear()
        self.actions.clear()
        self.rewards = [np.nan]
        self.dones = [False]

    def reset(self) -> None:
        """Restart buffers **and** reset the policy for a **new simulation**."""
        self.restart()
        self._policy = deepcopy(self._policy_backup)

    # ---------------------------------------------------------------- I/O
    def save(self, file: str) -> None:
        """Persist policy parameters to *file*."""
        self._policy.save(file=file)

    def load(self, file: str) -> None:
        """Load policy parameters from *file*."""
        self._policy.load(file=file)
