from __future__ import annotations

"""Episode utilities.

© Edgar Andrade 2024  
Email: edgar.andrade@urosario.edu.co

Helper functions to gather, process, and visualise data.

Main public API
---------------
- `Episode` : run one or many episodes, convert traces to `pandas.DataFrame`,
  render, or export to CSV.

Design highlights
-----------------
- Strict typing (`mypy --strict`‑clean).
- `@dataclass(slots=True)` for memory efficiency.
- Single‑responsibility helpers and explicit error types.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from time import sleep
from typing import Any, Callable, Deque, Generic, List, Optional, Tuple, TypeVar

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from gymnasium.utils.save_video import save_video
from tqdm.auto import tqdm

from agents.BaseAgent import Agent  # type: ignore
from interaction_utils.interpreters import id_state

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
S = TypeVar("S")  # state type
A = TypeVar("A")  # action type

StateInterpreter = Callable[[Any], S]


# ---------------------------------------------------------------------------
# Logging / verbosity
# ---------------------------------------------------------------------------
class Verbosity(IntEnum):
    NONE = 0
    SIM = 1
    EPISODE = 2
    ROUND = 3
    STEP = 4


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Episode(Generic[S, A]):
    """Run, record, and render episodes of a Gymnasium environment."""

    env: gym.Env
    env_name: str
    agent: Agent  # Generic protocol in your code‑base
    model_name: str
    num_rounds: int
    episode_id: int = 0
    state_interpreter: StateInterpreter = id_state
    sleep_time: float = 0.3

    # Internal state (initialised post‑construction)
    done: bool = field(init=False, default=False)
    T: int = field(init=False, default=0)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        obs, *_ = self.env.reset()
        state = self.state_interpreter(obs)
        self.initial_state: S = state
        self.agent.restart()
        self.agent.states.append(state)

    # ------------------------------------------------------------------ API
    def play_round(self, *, verbose: Verbosity = Verbosity.NONE, learn: bool = True) -> None:
        """Execute a single environment step."""

        # --- 1. Agent decision ------------------------------------------------
        try:
            action: A = self.agent.make_decision()
        except Exception as exc:  # pragma: no‑cover
            raise RuntimeError("Agent could not choose an action") from exc

        # --- 2. Environment transition --------------------------------------
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state: S = self.state_interpreter(obs)

        # --- 3. Book‑keeping --------------------------------------------------
        self.agent.actions.append(action)
        if hasattr(self.agent, "next_states"):
            self.agent.next_states.append(next_state)  # type: ignore[attr-defined]
        self.agent.rewards.append(float(reward))
        self.agent.dones.append(bool(done))

        if verbose >= Verbosity.ROUND:
            logger.info(
                "state=%s action=%s next_state=%s reward=%s done=%s",
                self.agent.states[-1],
                action,
                next_state,
                reward,
                done,
            )

        # --- 4. Agent learning ----------------------------------------------
        if learn:
            try:
                self.agent.update()  # type: ignore[arg-type]
            except Exception:
                # Fallback signature: update(next_state, reward, done)
                self.agent.update(next_state, reward, done)  # type: ignore[arg-type]

        # --- 5. Increment counters ------------------------------------------
        self.agent.states.append(next_state)
        self.T += 1
        self.done = done

    # ------------------------------------------------------------------ run
    def run(self, *, verbose: Verbosity = Verbosity.NONE, learn: bool = True) -> pd.DataFrame:
        """Run *one* episode for up to *num_rounds* timesteps."""

        self.reset()
        for t in range(self.num_rounds):
            if self.done:
                break
            if verbose >= Verbosity.EPISODE:
                logger.info("Round %d/%d", t + 1, self.num_rounds)
            self.play_round(verbose=verbose, learn=learn)
        return self._to_dataframe()

    # ---------------------------------------------------------------- reset
    def reset(self) -> S:
        """Reset environment **and** agent; return initial state."""

        obs, *_ = self.env.reset()
        state = self.state_interpreter(obs)
        self.agent.restart()

        if isinstance(self.agent.states, deque):
            self.agent.states.clear()
            self.agent.states.append(state)
        else:
            self.agent.states = [state]  # type: ignore[assignment]

        self.T = 0
        self.done = False
        return state

    # --------------------------------------------------------- public helpers
    def render(
        self,
        *,
        to_video: bool = False,
        folder: Optional[str] = None,
        verbose: Verbosity = Verbosity.NONE,
    ) -> None:
        """Live‑render current episode; optionally export to mp4."""

        if to_video:
            if folder is None:
                raise ValueError("folder path must be provided when to_video=True")
            if self.env.render_mode != "rgb_array":  # pragma: no‑cover
                raise RuntimeError("env.render_mode must be 'rgb_array' for video export")

        frames: List[np.ndarray] = []
        img = plt.imshow(np.zeros((2, 2)))

        for _ in range(self.num_rounds):
            if self.done:
                break
            frame = self.env.render()
            if isinstance(frame, np.ndarray):
                if to_video:
                    frames.append(frame)
                clear_output(wait=True)
                img.set_data(frame)
                plt.axis("off")
                display(plt.gcf())
            sleep(self.sleep_time)
            self.play_round(verbose=verbose, learn=False)

        if to_video and frames:
            save_video(frames, video_folder=folder, fps=int(1 / self.sleep_time))
        plt.close()

    def simulate(
        self,
        *,
        num_episodes: int = 1,
        save_path: Optional[str] = None,
        verbose: Verbosity = Verbosity.NONE,
        learn: bool = True,
        cool_down: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run *num_episodes* and concatenate the resulting traces."""

        data_frames: List[pd.DataFrame] = []
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            self.episode_id = ep
            if cool_down:
                self._apply_cool_down(cool_down, num_episodes, ep)
            data_frames.append(self.run(verbose=verbose, learn=learn))
        df = pd.concat(data_frames, ignore_index=True)
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df

    # ---------------------------------------------------------------- private
    def _trim_buffers(
        self,
    ) -> Tuple[List[S], List[A], List[float], List[bool], Optional[List[S]]]:
        """Return aligned, per‑episode buffers (state lists are one element longer)."""

        states: List[S] = self.agent.states[:-1]  # discard final s'
        length = min(self.T, len(states))
        actions: List[A] = self.agent.actions[-length:]
        rewards: List[float] = self.agent.rewards[-length:]
        dones: List[bool] = self.agent.dones[-length:]
        next_states: Optional[List[S]] = None
        if hasattr(self.agent, "next_states"):
            next_states = self.agent.next_states[-length:]  # type: ignore[attr-defined]

        assert len(actions) == len(states) == len(rewards) == len(dones)
        if next_states is not None:
            assert len(next_states) == len(states)
        return states, actions, rewards, dones, next_states

    def _to_dataframe(self) -> pd.DataFrame:
        states, actions, rewards, dones, next_states = self._trim_buffers()
        length = len(states)
        data: dict[str, Any] = {
            "model": [self.model_name] * length,
            "environment": [self.env_name] * length,
            "episode": [self.episode_id] * length,
            "round": list(range(length)),
            "state": states,
            "action": actions,
            "reward": rewards,
            "done": dones,
        }
        if next_states is not None:
            data["next_state"] = next_states
        return pd.DataFrame(data)

    def _apply_cool_down(self, cool_down: str, num_episodes: int, ep: int) -> None:
        if not cool_down.startswith("staircase-"):
            logger.warning("cool_down '%s' not recognised; ignored.", cool_down)
            return

        steps = int(cool_down.split("-", maxsplit=1)[1])
        bin_size = num_episodes / steps
        bin_idx = min(int(ep / bin_size), steps - 1)
        self.agent.epsilon = 1.0 - (bin_idx / steps)
        logger.info("Episode %d — epsilon set to %.3f", ep, self.agent.epsilon)
