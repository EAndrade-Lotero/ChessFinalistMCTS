# -*- coding: utf-8 -*-
"""
AlphaZero Monte-Carlo Tree Search
=========================================

Contents
--------
2. SearchTreeNode   – one node in the MCTS tree, with UCB support
3. GameSearchTree   – the search algorithm itself (uses `heapq`)

Requirements
------------
* numpy             – for `np.log` and `np.sqrt`
* a `game` object   – must implement: acciones, resultado, es_terminal, utilidad
"""

from __future__ import annotations

import heapq
import pydot

import numpy as np
from numpy.random import Generator, default_rng

from tqdm.auto import tqdm

from pprint import pprint
from itertools import count
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Set, Optional

from agents.base_classes import PolicyProtocol, GameProtocol, EncoderProtocol
from mcts.trees import PriorityQueue, NodeValue

import heapq
import itertools
from typing import Generic, TypeVar, Tuple

T = TypeVar("T")

# --------------------------------------------------------------------------- #
#                           2.  Search tree node                              #
# --------------------------------------------------------------------------- #
class SearchTreeNode:
    """
    A single node in the Monte-Carlo search tree.

    Parameters
    ----------
    state : Any
        Game-specific state representation.
    parent : SearchTreeNode | None
        Parent node; ``None`` for the root.
    action : Any | None
        Action that led from *parent* to *state* (``None`` at root).
    untried_actions : List[Any]
        List of untried actions from this node.
    value : NodeValue
        Current playout statistics.
    puct_constant : float
        Exploration factor *c* in UCB₁.
    """

    # ------------- construction ----------------------------------------- #
    def __init__(
        self,
        state: Any,
        player: str,
        parent: "SearchTreeNode | None",
        action: "Any | None",
        value: NodeValue,
        puct_constant: float,
    ) -> None:
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.value = value
        self.puct_constant = 1.0
        self.children = []
        self.finished = False
        
    # ------------- Monte-Carlo helpers ---------------------------------- #
    def backpropagate(self, result: int) -> Tuple[Any, NodeValue]:
        """
        Update statistics along the path back to the root.

        Returns
        -------
        (root_action, updated_root_value)
        """
        if self.player == "white":
            #If Player is white, black has just played
            if result == 1:
                #Black Lost
                pass
            elif result == -1:
                #Black Won
                self.value.reward -= result
        elif self.player == "black":
            #If Player is black, white has just played
            if result == 1:
                #White Won
                self.value.reward += result
            elif result == -1:
                #White Lost
                pass

        self.value.playouts += 1

        if self.depth() == 0:  # parent is the root
            return self.action, self.value
        else:
            # recurse until we reach depth 0
            return self.parent.backpropagate(result)  # type: ignore[arg-type]

    def puct(self, total_visits: int, q: float, v: float) -> float:
        """PUCT"""
        exploration = self.puct_constant * v * np.sqrt(total_visits) / (1 + self.value.playouts)
        return q + exploration

    def mean_value(self) -> float:
        return (
            self.value.reward / self.value.playouts if self.value.playouts > 0 else 0.0
        )

    # ------------- utility --------------------------------------------- #
    def depth(self) -> int:
        return 0 if self.parent is None else 1 + self.parent.depth()
    
    def get_action_history(self) -> List[str]:
        if self.parent is None:
            return []
        else:
            return self.parent.get_action_history() + [str(self.action)]

    def is_finished(self) -> bool:
        if self.finished:
            return True
        return np.all([child.finished for child in self.children])

    def __str__(self) -> str:
        root_flag = "--root--" if self.parent is None else ""
        s = f"State {root_flag}\n{self.state}\nDepth: {self.depth()}\n"
        s += f"Reward/Visits: {self.value}\n"
        if self.parent is not None:
            s += f"From action: {self.action}\nValue: {self.value}\n"
        return s


# --------------------------------------------------------------------------- #
#                       3.  MCTS                                              #
# --------------------------------------------------------------------------- #
class GameSearchTree:
    """
    Beam-width-limited Monte-Carlo Tree Search with an *ascending* heap frontier.

    Parameters
    ----------
        rng
        `numpy.random.Generator` used for sampling.  If *None*, the policy
        creates an independent generator via `default_rng()`.
    """

    def __init__(
        self,
        root: Any | SearchTreeNode,
        game: GameProtocol,
        sim_limit: int,
        puct_constant: float,
        n_iterations: int,
        value_network,
        policy_network,
        encoder: EncoderProtocol,
        rng: Optional[Generator] = None
    ) -> None:
        # Networks
        self.value_network = value_network
        self.policy_network = policy_network

        # Encoder
        self.encoder = encoder

        # Random number generator
        self.rng: Generator = rng or default_rng()

        # Hyper-parameters & helpers
        if not isinstance(game, GameProtocol):
            raise TypeError(
                "`game` must implement a Game Protocol."
            )
        self.game = game
        self.sim_limit = sim_limit
        self.puct_constant = puct_constant
        self.n_iterations = n_iterations

        # Global playout counter
        self.total_playouts: int = 0

        # Initialize the root's children
        if isinstance(root, SearchTreeNode):
            self.root = root
        else:
            try:
                self._expand_root(root)
            except Exception as e:
                """print(f'A game state was expected, but something went wrong!')"""
                raise e

    # ---------------- public helper to pick the best root move ----------- #
    def get_best_root_action(self) -> Any | None:
        """Pick the root's action with highest UCB"""

        # Values of all actions from the root
        value = self.value_network.predict(self.root.state)
        
        # Politica de todas las acciones desde la raiz
        probabilities = self.policy_network.predict(self.root.state)

        # Create priority queue with children
        children = PriorityQueue()
        for child in self.root.children:
            p = probabilities[child.action]
            puct = child.puct(self.total_playouts, value, p)
            children.push(puct, child)

        _, best_child = children.pop()
        if best_child is None:
            return None
        else:
            return best_child.action

    def get_child_with_highest_ucb(
        self, 
        node: SearchTreeNode, 
        skip_finished: Optional[bool] = True,
        skip_terminals: Optional[bool] = True,
    ) -> Any | None:
        """Pick the node's child with highest UCB"""
        multiplier = 1
        """multiplier = 1 if self.game.player(node.state) == 'white' else -1"""
        """print(f"{multiplier=}")"""
        best_ucb = -np.inf * multiplier
        matches = []

        # Create priority queue with children
        children = PriorityQueue()
        for child in node.children:
            ucb = child.ucb(self.total_playouts) * multiplier
            children.push(ucb, child)

        # Iterate to find child with best ucb with conditions
        child = None
        while child is None and children:
            best_ucb, child = children.pop()
            is_terminal = self.game.is_terminal(child.state)
            # if is_terminal and skip_terminals: 
            #     child = None
            #     continue
            if child.is_finished() and skip_finished:
                child = None
                continue

        return child

        #     ucb = child.ucb(self.total_playouts) * multiplier
        #     is_terminal = self.game.is_terminal(child.state)
        #     if is_terminal and skip_terminals: 
        #         continue
        #     elif not self.has_non_terminal_children(child):
        #         for grandchild in child.children:
        #             result = self.game.utility(grandchild.state)
        #             self.backpropagate(grandchild, result)
        #         continue
        #     elif multiplier == 1:
        #         if ucb > best_ucb:
        #             """print(f"{ucb=} --- {best_ucb}")"""
        #             best_ucb = ucb
        #             matches = []
        #     elif multiplier == -1:
        #         if ucb < best_ucb:
        #             best_ucb = ucb
        #             matches = []
        #     else:
        #         raise Exception('WWWWWTTTTTTTTFFFFFFF')
        #     if (ucb == best_ucb):
        #         """if (not is_terminal) or (not skip_terminals):"""
        #         matches.append(child)
        # if len(matches) == 0:
        #     raise Exception('WTF')
        #     return None
        # best_child = self.rng.choice(matches)
        # return best_child

    def has_non_terminal_children(self, node: SearchTreeNode) -> bool:
        non_terminal_children = [child for child in node.children if not self.game.is_terminal(child.state)]        
        if len(non_terminal_children) > 0:
            return True
        else:
            return False

    def get_root_child_from_action(self, action):
        for child in self.root.children:
            if action == child.action:
                return child
        return None

    # ---------------- tree expansion ------------------------------------ #
    def expand_node(self, node: SearchTreeNode) -> None:
        """Expand node with rollout policy"""

        assert(not node.is_fully_expanded()), f"Error: Node is fully expanded and cannot be expanded!\n{node}\n{node.get_action_history()}"
        assert(not node.finished), f"Error: Node is finished and cannot be expanded!\n{node}\n{node.get_action_history()}"

        # Choose an action according to rollout policy
        probabilities_untried_actions = self.rollout_policy.predict_in_list(node.state, node.untried_actions)
        action = self.rng.choice(node.untried_actions, p=probabilities_untried_actions)
        if isinstance(node.untried_actions, list):
            node.untried_actions.remove(action)
        else:
            raise Exception(f"Error: untried action should be a list (got {type(node.untried_actions)})")

        # Create new child
        new_state = self.game.result(node.state, action)

        # Find player
        player = self.game.player(new_state)

        untried_actions = self.game.actions(new_state)
        if len(untried_actions) > self.beam_width:
            untried_actions = self.rng.choice(untried_actions, self.beam_width, replace=False).tolist()

        child = SearchTreeNode(
            state=new_state,
            player= player,
            parent=node,
            action=action,
            untried_actions=untried_actions,
            value=NodeValue(0, 0),
            puct_constant=self.puct_constant
        )

        if self.game.is_terminal(child.state):
            child.finished = True
            result = self.game.utility(child.state)
            self.backpropagate(child, result)

        node.children.append(child)
        node.finished = node.is_finished()

    def _expand_root(self, state: Any) -> None:
        # Find Player
        player = self.game.player(state)

        # Get root actions
        actions = self.game.actions(state)
        if not actions:
            raise Exception('Error: No valid actions from root!')

        self._root_actions = actions

        # Root node
        self.root = SearchTreeNode(
            state=state,
            player=player,
            parent=None,
            action=None,
            value=NodeValue(0, 0),
            puct_constant=self.puct_constant,
        )

        for a in self._root_actions:
            new_state = self.game.result(state, a)
            # Find Player
            player = self.game.player(new_state)

            child = SearchTreeNode(
                state=new_state,
                player=player,
                parent=self.root, 
                action=a, 
                value=NodeValue(0, 0),
                puct_constant=self.puct_constant
            )

            if self.game.is_terminal(child.state):
                child.finished = True
                result = self.game.utility(child.state)
                self.backpropagate(child, result)

            self.root.children.append(child)
    
    # ---------------- selection & backup -------------------------------- #
    def select_ucb(self) -> SearchTreeNode | None:
        """Tree search strategy"""
        node = self.root
        best_node = self._recursive_select_ucb(node)
        return best_node

    def _recursive_select_ucb(self, node: SearchTreeNode) -> SearchTreeNode | None:
        """Recursive search"""

        best_child = self.get_child_with_highest_ucb(
            node=node, 
            skip_finished=True
        )
        if best_child is None:
            print(f"{node.state}")
            raise Exception(f"Ooops, didn't find usable best child from node\n{node}\nThe action history is:\n{node.get_action_history()}")

        if not best_child.is_fully_expanded():
            return best_child
        elif best_child.is_finished():
            return None
        return self._recursive_select_ucb(best_child)

    def backpropagate(self, node: SearchTreeNode, result: int) -> None:
        """Update stats and rebuild frontier priorities after each playout."""
        self.total_playouts += 1
        action, new_val = node.backpropagate(result)

    # ---------------- Policy rollout -------------------------------- #
    def make_rollout(self, node: SearchTreeNode) -> float:
        """Self-play using rollout policy"""
        counter = 0
        state = node.state.copy()
        while (counter < self.sim_limit) and (not self.game.is_terminal(state)):
            action = self.rollout_policy(state)
            state = self.game.result(state, action)
            counter += 1
        utility = self.game.utility(state)
        if utility == 0: utility = -1
        return utility

    # ---------------- The pipeline -------------------------------------- #
    def make_decision(self) -> Any | None:

        if self.root.is_finished():
            return self.get_best_root_action()
        
        counter = 0
        while counter < self.n_iterations:

            # Step 1: Node selection
            node = self.select_ucb()
            if node is None:
                raise Exception(f"Ooops, no ucb selection from node\n{node}\n{node.get_action_history()}")
            elif node.is_fully_expanded():
                raise Exception(f"Ooops, no expansion from node\n{node}\n{node.get_action_history()}")
            elif node.is_finished():
                raise Exception(f"Ooops, finished node not for expansion\n{node}\n{node.get_action_history()}")

            # Step 2: Expansion
            self.expand_node(node)

            # Step 3: Rollout
            rollout_result = self.make_rollout(node)

            # Step 4: Backpropagate
            self.backpropagate(node, rollout_result)

            counter += 1

        # Get best action
        return self.get_best_root_action()  
  
    # ---------------- visualization helpers -------------------------------- #   
    def to_pydot(self) -> pydot.Dot:
        """Return a pydot graph representing the current search tree."""

        # ------------------------------------------------------------------ #
        # 1.  User-facing label function (make it a real Callable for mypy)
        # ------------------------------------------------------------------ #
        def label_fn(node: SearchTreeNode) -> str:
            state = node.state
            print(state)
            player = self.game.player(state)
            state_, player = self.encoder.encode_obs(state)
            print(state_)
            v = self.value_network(state_)
            q = self.policy_network(state_)[node.action]

            msg = f"Value={node.value}\n"
            msg += f"To play={player}\n"
            msg += f"PUCT={node.puct(self.total_playouts, q, v):.2f}\n"
            msg += f"Fully expanded={node.is_fully_expanded()}\n"
            msg += f"Finished={node.finished}\n"
            msg += f"Terminal={self.game.is_terminal(node.state)}"
            if node.action is None:
                msg = "root\n" + msg
            else: 
                msg = f"{node.action}\n" + msg
            return msg

        # ------------------------------------------------------------------ #
        # 2. Unique string id for each SearchTreeNode
        # ------------------------------------------------------------------ #
        _id_counter = count()
        id_map: Dict[SearchTreeNode, str] = {}

        def get_id(node: SearchTreeNode) -> str:
            """Return a stable string id for *node* (create on first call)."""
            if node not in id_map:
                id_map[node] = f"n{next(_id_counter)}"
            return id_map[node]

        # ------------------------------------------------------------------ #
        # 3.  Initialise the pydot graph
        # ------------------------------------------------------------------ #
        graph: pydot.Dot = pydot.Dot(graph_type="digraph", rankdir="TB")

        # ------------------------------------------------------------------ #
        # 3.  DFS adding edges/nodes
        # ------------------------------------------------------------------ #
        stack: List[SearchTreeNode] = [self.root]

        while stack:
            node: SearchTreeNode = stack.pop()

            node_id: str = get_id(node)
            graph.add_node(
                pydot.Node(node_id, label=label_fn(node), shape="box")
            )

            for child in node.children:
                child_id: str = get_id(child)
                graph.add_edge(pydot.Edge(node_id, child_id))
                stack.append(child)

        return graph