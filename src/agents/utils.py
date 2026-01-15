from __future__ import annotations

# import chess
# import cairosvg

import numpy as np

from chess import Board, Move


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
    rangos = np.array([0, 8, 16, 44])
    n_actions: int = 44  # 44 possible actions
    dict_codificacion_rey = {
        (-1, -1): 0,
        (-1, 0): 1,
        (-1, 1): 2,
        (0, -1): 3,
        (0, 1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7,
    }
    list_codificacion_rey = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)
    ]
    dict_codificacion_torre = {
        (0, -7): 0,
        (0, -6): 1,
        (0, -5): 2,
        (0, -4): 3,
        (0, -3): 4,
        (0, -2): 5,
        (0, -1): 6,
        (0, 1): 7,
        (0, 2): 8,
        (0, 3): 9,
        (0, 4): 10,
        (0, 5): 11,
        (0, 6): 12,
        (0, 7): 13,
        (7, 0): 14,
        (6, 0): 15,
        (5, 0): 16,
        (4, 0): 17,
        (3, 0): 18,
        (2, 0): 19,
        (1, 0): 20,
        (-1, 0): 21,
        (-2, 0): 22,
        (-3, 0): 23,
        (-4, 0): 24,
        (-5, 0): 25,
        (-6, 0): 26,
        (-7, 0): 27,
    }
    list_codificacion_torre = [
        (0, -7), (0, -6), (0, -5), (0, -4), (0, -3), (0, -2), (0, -1), 
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
        (7, 0), (6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), 
        (-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0), (-6, 0), (-7, 0)
    ]

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
        self.action_space = spaces.Discrete(self.n_actions)

    # ----------------------------- State Encoding -------------------------- #

    def encode_obs(self, board: Board) -> np.ndarray:
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

        # # 1) Build SVG string from python-chess
        # svg_str = chess.svg.board(board, size=size)

        # # 2) Convert SVG → PNG bytes
        # png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))

        # # 3) Load PNG bytes into a Pillow image and convert mode
        # img = Image.open(BytesIO(png_bytes)).convert(mode)

        # # 4) Convert to NumPy array
        # arr = np.asarray(img)

        # # 5) normalization
        # arr = (arr.astype(np.float32) / 255.0)

        # return arr

        tablero = str(board)
        t1 = tablero.split('\n')
        t1 = [linea.replace('k', '1') for linea in t1]
        t1 = [linea.replace('K', '2') for linea in t1]
        t1 = [linea.replace('R', '3') for linea in t1]
        t1 = [linea.replace('.', '0') for linea in t1]
        t1 = [linea.split(' ') for linea in t1]
        t1 = [[int(x) for x in linea] for linea in t1]
        t1 = np.array(t1)
        return t1

    # ----------------------------- Action Encoding ------------------------- #

    def encode_action(self, board: Board, action: Any) -> int:
        """
        Encode a domain action into an int usable by Gym.

        If your domain actions are already integers, this is identity.
        Otherwise, you might map them into an index.
        """
        if not isinstance(action, Move):
            raise ValueError(f"Cannot encode non-integer action: {action}")
        if not isinstance(board, Board):
            raise ValueError(f"board should be of type Board (got {type(board)} instead.")
    
        coded_board = self.encode_obs(board)
        salida, llegada = self.casillas_desde_hasta(action)
        pieza = coded_board[salida]
        diferencia = np.array(llegada) - np.array(salida)
        one_hot_rey_negro = np.zeros(8)
        one_hot_rey_blanco = np.zeros(8)
        one_hot_torre = np.zeros(28)

        if pieza == 1:
            accion_rey = self.dict_codificacion_rey[tuple(diferencia)]
            one_hot_rey_negro[accion_rey] = 1
        elif pieza == 2:
            accion_rey = self.dict_codificacion_rey[tuple(diferencia)]
            one_hot_rey_blanco[accion_rey] = 1
        elif pieza == 3:
            accion_torre = self.dict_codificacion_torre[tuple(diferencia)]
            one_hot_torre[accion_torre] = 1
        else:
            raise ValueError(f"Pieza incorrecta. Se esperaba 1, 2 o 3 (pero se obtuvo {pieza})")

        one_hot = np.concat([one_hot_rey_negro, one_hot_rey_blanco, one_hot_torre])
        return one_hot
        
    def casillas_desde_hasta(self, action: Move) -> Tuple[int, int]:
        print(f"{action=}")

        coded_index = action.from_square
        print(f"{coded_index=}")
        from_index_pair = np.unravel_index(coded_index, (8,8))
        fila, columna = from_index_pair
        fila = 7 - fila
        from_index_pair = (fila, columna)

        coded_index = action.to_square
        print(f"{coded_index=}")
        to_index_pair = np.unravel_index(coded_index, (8,8))
        fila, columna = to_index_pair
        fila = 7 - fila
        to_index_pair = (fila, columna)

        casillas = [from_index_pair, to_index_pair]
        return casillas



    def decode_action(self, board: Board, action: Any) -> Any:
        """
        Decode a Gym action (int index) into a domain action.

        By default, returns valid_actions[action].
        """
        bin_idx = np.digitize(action, self.rangos)
        obs = self.encode_obs(board)
        fila, columna = np.where(obs == bin_idx)
        assert(0 <= columna[0] < 8)
        assert(0 <= fila[0] < 8)
        casilla_desde = f"{chr(columna[0] + 97)}{8 - fila[0]}"
        offset = self.rangos[bin_idx - 1]
        indice_pieza = action - offset 
        if bin_idx in [1, 2]:
            fila_mas, columna_mas = self.list_codificacion_rey[indice_pieza]
        elif bin_idx in [3]:
            fila_mas, columna_mas = self.list_codificacion_torre[indice_pieza]
        else:
            print(bin_idx)
            raise ValueError
        casilla_hasta_ = (fila + fila_mas, columna + columna_mas)
        fila, columna = casilla_hasta_
        print(f"{casilla_hasta_=}")
        assert(0 <= columna < 8)
        assert(0 <= fila < 8)
        casilla_hasta =  f"{chr(columna[0] + 97)}{8 - fila[0]}"
        print(casilla_hasta)
        algebraico = f"{casilla_desde}{casilla_hasta}"
        return Move.from_uci(algebraico)
    
