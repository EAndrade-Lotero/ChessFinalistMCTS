# -*- coding: utf-8 -*-
"""
KRK ‒ King-and-Rook vs King mini-chess environment
==================================================

This helper class wraps *python-chess* so you can experiment with the
King-and-Rook vs King end-game (hence **K-R-K**).  It exposes a very small,
gym-like interface:

    ▸ `initial_state`      –   starting board (python-chess.Board)
    ▸ `render()`           –   SVG visualisation inside a notebook
    ▸ `player()`           –   whose turn is it? ('white' / 'black')
    ▸ `actions()`          –   legal moves in SAN or chess.Move list
    ▸ `move_manual()`      –   apply a SAN string *if* it is legal
    ▸ `result()`           –   successor state after a move
    ▸ `is_terminal()`      –   has the game ended?
    ▸ `utility()`          –   ±1 for check-mate, 0 for draw
    ▸ `announce_result()`  –   pretty-print winner / draw

Author: <your-name>
"""

from __future__ import annotations

import copy
# import chess
# import chess.svg
# import cairosvg

import games.chambon_chess as chess

import numpy as np

from PIL import Image
from io import BytesIO
from IPython.display import SVG, display

from agents.base_classes import GameProtocol


class KRK(GameProtocol):
    """
    Create a King-and-Rook vs King end-game playground.

    Parameters
    ----------
    side_to_move : {'white', 'black'}, default='white'
        Which colour plays first.
    start_position : int, 1‒9, default=1
        Index of a predefined KRK position (see `_POSITIONS`).

    Raises
    ------
    ValueError
        If `side_to_move` or `start_position` is invalid.
    """

    # Pre-canned positions in FEN (without the side-to-move bit).
    _POSITIONS = {
        1: "8/8/8/8/8/2R5/2K5/k7",
        2: "8/8/8/8/8/2R5/k1K5/8",
        3: "8/8/8/8/8/8/1k1K4/2R5",
        4: "2R5/8/8/8/8/k7/2K5/8",
        5: "8/8/8/5k2/8/8/6K1/7R",
        6: "8/8/8/4k3/8/8/6K1/7R",
        7: "8/5k2/8/8/6K1/8/8/2R5",
        8: "8/8/2R2k2/8/6K1/8/8/8",
        9: "7k/8/8/6K1/8/8/8/7R",
    }

    def __init__(self, side_to_move: str = "white", start_position: int = 1) -> None:
        # ----------- validate inputs --------------------------------- #
        side_to_move = side_to_move.lower()
        if side_to_move not in ("white", "black"):
            raise ValueError("`side_to_move` must be 'white' or 'black'.")

        if start_position not in self._POSITIONS:
            raise ValueError(
                f"`start_position` must be between 1 and {len(self._POSITIONS)}."
            )
        # -------------------------------------------------------------- #

        fen_suffix = " w" if side_to_move == "white" else " b"
        self.initial_state: chess.Board = chess.Board(
            self._POSITIONS[start_position] + fen_suffix
        )

    # --------------------------------------------------------------------- #
    #                               Helpers                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def render(board: chess.Board) -> None:
        """
        Display a small SVG of the board (for Jupyter / Colab notebooks).

        Parameters
        ----------
        board : chess.Board
        """
        display(SVG(chess.svg.board(board, size=300)))

    @staticmethod
    def image_to_numpy(board: chess.Board) -> np.ndarray:
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
        size = 300
        mode = 'RGB'

        # 1) Build SVG string from python-chess
        svg_str = chess.svg.board(board, size=size)

        # 2) Convert SVG → PNG bytes
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))

        # 3) Load PNG bytes into a Pillow image and convert mode
        img = Image.open(BytesIO(png_bytes)).convert(mode)

        # 4) Convert to NumPy array
        arr = np.asarray(img)

        # 5) normalization
        return (arr.astype(np.float32) / 255.0)

    @staticmethod
    def player(board: chess.Board) -> str:
        """
        Return the colour whose turn it is.

        Returns
        -------
        'white' | 'black'
        """
        return "white" if board.turn else "black"

    # --------------------------------------------------------------------- #
    #                          Environment API                              #
    # --------------------------------------------------------------------- #
    @staticmethod
    def actions(board: chess.Board) -> list[chess.Move]:
        """
        Enumerate all legal moves from the given position.

        Returns
        -------
        list[chess.Move]
        """
        return list(board.legal_moves)

    @staticmethod
    def move_manual(board: chess.Board, san: str) -> chess.Board:
        """
        Manually apply a SAN-formatted move if it is legal.

        Parameters
        ----------
        board : chess.Board
        san   : str
            Move in Standard Algebraic Notation.

        Returns
        -------
        chess.Board
            A **new** board with the move played.

        Raises
        ------
        ValueError
            If the SAN string is not legal in this position.
        """
        try:
            move = board.parse_san(san)
        except ValueError as exc:  # invalid SAN (e.g., typo)
            raise ValueError("Invalid SAN string.") from exc

        if move not in board.legal_moves:
            raise ValueError("Move is not legal in the given position.")

        new_board = copy.deepcopy(board)
        new_board.push(move)
        return new_board

    @staticmethod
    def result(board: chess.Board, move: chess.Move) -> chess.Board:
        """
        Return the successor state after a **legal** move.

        The caller must ensure `move` is legal; otherwise python-chess will
        raise an exception.

        Parameters
        ----------
        board : chess.Board
        move  : chess.Move

        Returns
        -------
        chess.Board
        """
        new_board = copy.deepcopy(board)
        new_board.push(move)
        return new_board

    # ------------------------------------------------------------------ #
    #                      Terminal detection & reward                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_terminal(board: chess.Board) -> bool:
        """
        Test whether the position is an end-state (mate, stalemate, etc.).
        """
        return board.outcome() is not None

    @staticmethod
    def utility(board: chess.Board) -> int | None:
        """
        Compute a simple utility value for terminal positions.

        +1 : the *side that just moved* delivers mate  
        -1 : the *side to move* is mated, or any draw (stalemate, 50-move rule, insufficient material)

        Returns
        -------
        int | None
            ±1 or 0 for terminal boards, otherwise None.
        """
        outcome = board.outcome()
        if outcome is None:
            return 0

        if outcome.winner is None:                # draw
            return -1
        return 1 if outcome.winner else -1

    # ------------------------------------------------------------------ #
    #                        Convenience printing                        #
    # ------------------------------------------------------------------ #
    def announce_result(self, board: chess.Board) -> None:
        """
        Pretty-print the winner (or draw) if the game is over.
        """
        if not self.is_terminal(board):
            print("Game still in progress.")
            return

        outcome = board.outcome()
        if outcome.winner is None:
            print("Game over. Draw!")
        elif outcome.winner:  # True → White
            print("Game over. White wins!")
        else:
            print("Game over. Black wins!")
