from __future__ import annotations

import chess
import cairosvg
import numpy as np

from PIL import Image
from io import BytesIO
from gymnasium import spaces
from typing import Any, Sequence, Tuple, Protocol

from agents.base_classes import EncoderProtocol

# --------------------------------------------------------------------------- #
#                              Chess Encoder                                 #
# --------------------------------------------------------------------------- #

class ChessEncoder(EncoderProtocol):
    """
    A Chess encoder that:
      - treats states as numpy arrays (float32),
      - maps Gym discrete actions (int indices) to domain actions.

    Parameters
    ----------
    n_actions : int
        Number of possible actions in the game.
    obs_shape : Tuple[int, ...]
        Shape of the numpy array representing observations.
    obs_low : float, default=-1.0
        Minimum value for observation space.
    obs_high : float, default=1.0
        Maximum value for observation space.
    """

    def __init__(
        self,
    ) -> None:
        # Define observation space (continuous)
        obs_low: float = -1.0
        obs_high: float = 1.0
        obs_shape: Tuple[int, ...] = (84, 84)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )

        # Define action space (discrete)
        n_actions: int = 64 * 63
        self.action_space = spaces.Discrete(n_actions)

    # ----------------------------- State Encoding -------------------------- #

    def encode_obs(self, board: chess.Board) -> np.ndarray:
        """
        Return board as a NumPy array from SVG image.

        Parameters
        ----------
        board : chess.Board

        Returns
        -------
        np.ndarray
            Shape (size, size, C) for RGB/RGBA or (size, size) for 'L'.
        """
        size = 84
        mode = 'L'

        # 1) Build SVG string from python-chess
        svg_str = chess.svg.board(board, size=size)

        # 2) Convert SVG → PNG bytes
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))

        # 3) Load PNG bytes into a Pillow image and convert mode
        img = Image.open(BytesIO(png_bytes)).convert(mode)

        # 4) Convert to NumPy array
        arr = np.asarray(img)

        # 5) normalization
        arr = (arr.astype(np.float32) / 255.0)

        return arr

    # ----------------------------- Action Encoding ------------------------- #

    def encode_action(self, action: Any) -> int:
        """
        Encode a domain action into an int usable by Gym.

        If your domain actions are already integers, this is identity.
        Otherwise, you might map them into an index.
        """
        if isinstance(action, (int, np.integer)):
            return int(action)
        raise ValueError(f"Cannot encode non-integer action: {action}")

    def decode_action(self, action: int, valid_actions: Sequence[Any]) -> Any:
        """
        Decode a Gym action (int index) into a domain action.

        By default, returns valid_actions[action].
        """
        if not (0 <= action < len(valid_actions)):
            raise ValueError(f"Action {action} out of bounds for {len(valid_actions)} choices")
        return valid_actions[action]
    
