from __future__ import annotations

# import chess
# import cairosvg

import torch
import numpy as np

from chess import Board, Move


from PIL import Image
from io import BytesIO
from gymnasium import spaces
from typing import (
    Any, Sequence, Tuple, 
    Protocol, Union, List, Dict
)

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
    dict_pieces = {
        1: 'k',  # black king
        2: 'K',  # white king
        3: 'R',  # white rook
        0: 1,
    }
    rangos = np.array([0, 8, 16, 30])
    n_actions: int = 30  # 30 possible actions
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

    def __init__(
        self,
    ) -> None:
        # Define observation space (continuous)
        obs_low: float = -np.inf
        obs_high: float = np.inf
        obs_shape: Tuple[int, ...] = (65, )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )

        # Define action space (discrete)
        self.action_space = spaces.Discrete(self.n_actions)

    # ----------------------------- State Encoding -------------------------- #

    def encode_obs(self, board: Board) -> Tuple[np.ndarray, str]:
        """
        Return board as a NumPy array from chess Board.
        """
        tablero = str(board)
        
        t1 = tablero.split('\n')
        t1 = [linea.replace('k', '1') for linea in t1]
        t1 = [linea.replace('K', '2') for linea in t1]
        t1 = [linea.replace('R', '3') for linea in t1]
        t1 = [linea.replace('.', '0') for linea in t1]
        t1 = [linea.split(' ') for linea in t1]
        t1 = [[int(x) for x in linea] for linea in t1]
        t1 = np.array(t1).flatten()
        #t1 = _norm_array(t1)
        turn = 'w' if board.turn else 'b'
        return (t1, turn)
    
    @staticmethod
    def _norm_array(x):
        x = x.astype(np.float32)
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
        z = np.divide(x - mu, sigma, out=np.zeros_like(x), where=sigma != 0)
        return z
    
    def to_array(self, observation: Tuple[np.ndarray, str]) -> torch.Tensor:
        array, player = observation

        if player == 'w':
            array = np.concat([array, np.ones(1)])
        else:
            array = np.concat([array, np.zeros(1)])
        return ChessEncoder._norm_array(array)
    
    def decode_obs(self, observation: Tuple[np.ndarray, str]) -> Board:
        """
        Decode a NumPy array observation back into a chess.Board.

        Parameters
        ----------
        observation : np.ndarray
            Observation array.

        Returns
        -------
        chess.Board
            Decoded board state.
        """
        def process_row(row: np.ndarray) -> str:
            row_string = ''
            sum_empty = 0
            for x in row:
                if x not in [0, 1, 2, 3]:
                    raise ValueError(f"Invalid value {x} in observation array.")
                val = self.dict_pieces[x]
                if val == 1:
                    sum_empty += 1
                else:
                    if sum_empty > 0:
                        row_string += str(sum_empty) 
                        sum_empty = 0
                    row_string += val
            if sum_empty > 0:
                row_string += str(sum_empty) 
            return row_string
        board, player = observation
        board = board.reshape((8,8))
        t1 = [process_row(row) for row in board]
        fen_suffix = f" {player}"
        board = '/'.join(t1) + fen_suffix
        return Board(board)
    
    # ----------------------------- Action Encoding ------------------------- #

    def encode_action(self, board: Union[Board, Tuple[np.ndarray, str]], action: Any) -> int:
        """
        Encode a domain action into an int usable by Gym.

        If your domain actions are already integers, this is identity.
        Otherwise, you might map them into an index.
        """
        if not isinstance(action, Move):
            raise ValueError(f"Action should be a move: {action}")

        if isinstance(board, tuple):
            board = self.decode_obs(board)
        if not isinstance(board, Board):
            raise ValueError(f"board should be of type Board (got {type(board)} instead.")
    
        coded_board, player = self.encode_obs(board)
        coded_board = coded_board.reshape((8,8))
        # print(f'encode_action - coded_board\n {coded_board}')
        salida, llegada = self.casillas_desde_hasta(action)
        pieza = coded_board[salida]
        diferencia = np.array(llegada) - np.array(salida)
        diferencia = tuple(diferencia)
        one_hot_rey_negro = np.zeros(8)
        one_hot_rey_blanco = np.zeros(8)
        one_hot_torre = np.zeros(14)

        if pieza == 1:
            accion_rey = self.dict_codificacion_rey[tuple(diferencia)]
            one_hot_rey_negro[accion_rey] = 1
        elif pieza == 2:
            accion_rey = self.dict_codificacion_rey[tuple(diferencia)]
            one_hot_rey_blanco[accion_rey] = 1
        elif pieza == 3:
            dict_codificacion_torre = self._get_dict_codificacion_torre(action)
            # print(f'dict_codificacion_torre\n {dict_codificacion_torre}')
            accion_torre = dict_codificacion_torre[diferencia]
            # print(f'accion_torre\n {accion_torre}')
            one_hot_torre[accion_torre] = 1
        else:
            raise ValueError(f"Pieza incorrecta. Se esperaba 1, 2 o 3 (pero se obtuvo {pieza})")

        one_hot = np.concat([one_hot_rey_negro, one_hot_rey_blanco, one_hot_torre])
        return one_hot
        
    def casillas_desde_hasta(self, action: Move) -> Tuple[int, int]:
        coded_index = action.from_square
        from_index_pair = np.unravel_index(coded_index, (8,8))
        fila, columna = from_index_pair
        fila = 7 - fila
        from_index_pair = (fila, columna)

        coded_index = action.to_square
        to_index_pair = np.unravel_index(coded_index, (8,8))
        fila, columna = to_index_pair
        fila = 7 - fila
        to_index_pair = (fila, columna)

        casillas = [from_index_pair, to_index_pair]
        return casillas

    def filacol_a_algebraico(self, fila: int, columna: int) -> str:
        return f"{chr(columna + 97)}{8 - fila}"
    
    def algebraico_a_filacol(self, algebraico: str) -> Tuple[int, int]:
        columna = ord(algebraico[0]) - 97
        fila = 8 - int(algebraico[1])
        return fila, columna
    
    def decode_action(self, board: Board, action: int) -> Move:
        """
        Decode a Gym action (int index) into a domain action.

        By default, returns valid_actions[action].
        """
        assert(action < 30), f"Error: action {action} is not valid. It should be less than 30."
        
        pieza_a_mover = np.digitize(action, self.rangos)
        obs, player = self.encode_obs(board)
        obs = obs.reshape((8,8))
        
        # print(f"{pieza_a_mover=}")
        # print(f"{self.rangos=}")
        # print(obs)
        # print(f"{np.where(obs == pieza_a_mover)=}")
        
        fila, columna = np.where(obs == pieza_a_mover)
        assert(0 <= columna[0] < 8)
        assert(0 <= fila[0] < 8)
        casilla_desde_tuple = (fila[0], columna[0])
        casilla_desde = self.filacol_a_algebraico(*casilla_desde_tuple)
        
        # print(f"{casilla_desde=}")
        
        offset = self.rangos[pieza_a_mover - 1]
        indice_pieza = action - offset
        
        # print(f"{offset=}")
        # print(f"{indice_pieza=}")
        # print(f"{ChessEncoder._get_list_acciones_torre(casilla_desde_tuple)=}")
        if pieza_a_mover in [1, 2]:
            fila_mas, columna_mas = self.list_codificacion_rey[indice_pieza]
        elif pieza_a_mover in [3]:
            list_codificacion_torre = self._get_list_acciones_torre(casilla_desde_tuple)
            fila_mas, columna_mas = list_codificacion_torre[indice_pieza]
        else:
            # print(pieza_a_mover)
            raise ValueError
        casilla_hasta_ = (fila + fila_mas, columna + columna_mas)
        fila, columna = casilla_hasta_
        assert(0 <= columna < 8), f"{columna=}"
        assert(0 <= fila < 8), f"{fila=}"
        casilla_hasta = self.filacol_a_algebraico(fila[0], columna[0])
        algebraico = f"{casilla_desde}{casilla_hasta}"
        return Move.from_uci(algebraico)
    
    @staticmethod
    def _get_list_acciones_torre(casilla_desde: Tuple[int, int]) -> List[Tuple[int,int]]:
        i, j = casilla_desde # fila, columna
        x_list = [(k,0) for k in range(-i,8-i)]
        y_list = [(0,k) for k in range(-j,8-j)]
        list_acciones = x_list + y_list
        list_acciones = [pareja for pareja in list_acciones if pareja != (0,0)]
        return list_acciones

    def _get_dict_codificacion_torre(self, action: Move) -> Dict[Tuple[int, int], int]:
        casilla_desde, casilla_hasta = self.casillas_desde_hasta(action)
        list_acciones = self._get_list_acciones_torre(casilla_desde)       
        dict_codificacion_torre = {casilla:i for i, casilla in enumerate(list_acciones)}
        return dict_codificacion_torre