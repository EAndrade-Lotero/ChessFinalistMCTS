# -*- coding: utf-8 -*-
"""
Beam-width–limited Monte-Carlo Tree Search
=========================================

Contents
--------
1. NodeValue        – cumulative reward / simulations container
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

from pprint import pprint
from itertools import count
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Set, Optional

from agents.base_classes import PolicyProtocol, GameProtocol

# --------------------------------------------------------------------------- #
#                           1.  Node statistics                               #
# --------------------------------------------------------------------------- #
class NodeValue:
    """Holds cumulative **reward** and number of **playouts** for a node."""

    def __init__(self, reward: float, playouts: int) -> None:
        self.reward: float = reward
        self.playouts: int = playouts

    def __str__(self) -> str:  # For quick debugging prints
        return f"{self.reward}/{self.playouts}"


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
    ucb_constant : float
        Exploration factor *c* in UCB₁.
    """

    # ------------- construction ----------------------------------------- #
    def __init__(
        self,
        state: Any,
        parent: "SearchTreeNode | None",
        action: "Any | None",
        untried_actions: List[Any],
        value: NodeValue,
        ucb_constant: float
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.untried_actions = untried_actions
        self.value = value
        self.ucb_constant = ucb_constant
        self.children = []
        
    # ------------- Monte-Carlo helpers ---------------------------------- #
    def backpropagate(self, result: int) -> Tuple[Any, NodeValue]:
        """
        Update statistics along the path back to the root.

        Returns
        -------
        (root_action, updated_root_value)
        """
        self.value.reward += result
        self.value.playouts += 1

        if self.depth() == 0:  # parent is the root
            return self.action, self.value
        else:
            # recurse until we reach depth 0
            return self.parent.backpropagate(result)  # type: ignore[arg-type]

    def ucb(self, total_visits: int) -> float:
        """Upper-Confidence Bound value used for selection."""
        exploration = (
            np.sqrt(np.log(total_visits + 1) / (self.value.playouts + 1))
            if total_visits
            else 1.0
        )
        return self.mean_value() + self.ucb_constant * exploration

    def mean_value(self) -> float:
        return (
            self.value.reward / self.value.playouts if self.value.playouts else 0.0
        )

    # ------------- utility --------------------------------------------- #
    def depth(self) -> int:
        return 0 if self.parent is None else 1 + self.parent.depth()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
        
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
        state: Any,
        game: GameProtocol,
        rollout_policy: PolicyProtocol,
        sim_limit: int,
        beam_width: int,
        ucb_constant: float,
        n_iterations: int,
        rng: Optional[Generator] = None
    ) -> None:
        # Random number generator
        self.rng: Generator = rng or default_rng()

        # Hyper-parameters & helpers
        if not isinstance(game, GameProtocol):
            raise TypeError(
                "`game` must implement a Game Protocol."
            )
        self.game = game

        if not isinstance(rollout_policy, PolicyProtocol):
            raise TypeError(
                "`game` must implement a Policy Protocol."
            )
        self.rollout_policy = rollout_policy

        self.sim_limit = sim_limit
        self.beam_width = beam_width
        self.ucb_constant = ucb_constant
        self.n_iterations = n_iterations

        # Global playout counter
        self.total_playouts: int = 0

        # Initialize the root's children
        self._expand_root(state)

        # Frontier implemented with `heapq`
        self._frontier: List[Tuple[float, int, SearchTreeNode]] = []
        self._counter = 0  # tie-breaker for equal priorities

    # ---------------- public helper to pick the best root move ----------- #
    def best_root_action(self) -> Any:
        """Return the root-level action with **highest average reward**."""
        best_mean, best_action_str = -float("inf"), None
        for a_str, v in self.action_stats.items():
            mean = v.reward / v.playouts if v.playouts else 0.0
            if mean > best_mean:
                best_mean, best_action_str = mean, a_str
        # Locate the actual action object by string comparison
        matches = [a for a in self._root_actions if str(a) == best_action_str]
        assert matches, "Inconsistent state: action string not found."
        return matches[0]

    # ---------------- tree expansion ------------------------------------ #
    def expand_node(self, node: SearchTreeNode) -> None:
        """Expand node with rollout policy"""

        assert(not node.is_fully_expanded()), f"Error: Node cannot be expanded!\n{node}"

        # Choose an action according to rollout policy
        probabilities_untried_actions = self.rollout_policy.predict_in_list(node.state, node.untried_actions)
        action = self.rng.choice(node.untried_actions, p=probabilities_untried_actions)
        if isinstance(node.untried_actions, list):
            node.untried_actions.remove(action)
        else:
            raise Exception(f"Error: untried action should be a list (got {type(node.untried_actions)})")

        # Create new child
        new_state = self.game.result(node.state, action)

        untried_actions = self.game.actions(new_state)
        if len(untried_actions) > self.beam_width:
            untried_actions = self.rng.choice(untried_actions, self.beam_width, replace=False).tolist()

        child = SearchTreeNode(
            state=new_state,
            parent=node,
            action=action,
            untried_actions=untried_actions,
            value=NodeValue(0, 0),
            ucb_constant=self.ucb_constant
        )

        if self.game.is_terminal(child.state):
            result = self.game.utility(child.state)
            self.backpropagate(child, result)

        node.children.append(child)

    def _expand_root(self, state: Any) -> None:

        # Get root actions
        actions = self.game.actions(state)
        if not actions:
            raise Exception('Error: No valid actions from root!')

        if len(actions) > self.beam_width:
            actions = self.rng.choice(actions, self.beam_width, replace=False)
        self._root_actions = actions

        # Root node
        self.root = SearchTreeNode(
            state=state,
            parent=None,
            action=None,
            untried_actions=[],
            value=NodeValue(0, 0),
            ucb_constant=self.ucb_constant,
        )

        for a in self._root_actions:
            new_state = self.game.result(state, a)

            untried_actions = self.game.actions(new_state)
            if len(untried_actions) > self.beam_width:
                untried_actions = self.rng.choice(untried_actions, self.beam_width, replace=False)

            child = SearchTreeNode(
                state=new_state, 
                parent=self.root, 
                action=a, 
                untried_actions=untried_actions,
                value=NodeValue(0, 0),
                ucb_constant=self.ucb_constant
            )

            if self.game.is_terminal(child.state):
                result = self.game.utility(child.state)
                self.backpropagate(child, result)

            self.root.children.append(child)
    
    # ---------------- selection & backup -------------------------------- #
    def select_ucb(self) -> SearchTreeNode | None:
        """Pick the node's child with highest UCB"""
        best_ucb = -np.inf
        self._frontier = []
        self._push_node(self.root, self.root.ucb(self.total_playouts))
        matches = []
        while len(self._frontier) > 0:
            node = self._pop_best_node()
            for child in node.children:
                if not self.game.is_terminal(child.state):
                    ucb = child.ucb(self.total_playouts)
                    if not child.is_fully_expanded():
                        if ucb > best_ucb:
                            best_ucb = ucb
                            matches = []
                        if ucb == best_ucb:
                            matches.append(child)
                    self._push_node(child, child.ucb(self.total_playouts))
        if len(matches) == 0:
            return None
        return self.rng.choice(matches)

    def get_best_root_action(self) -> Any | None:
        """Pick the root's action with highest UCB"""
        best_ucb = -np.inf
        matches = []
        for child in self.root.children:
            ucb = child.ucb(self.total_playouts)
            if ucb > best_ucb:
                best_ucb = ucb
                matches = []
            if ucb == best_ucb:
                matches.append(child)
        if len(matches) == 0:
            return None
        best_child = self.rng.choice(matches)
        return best_child.action

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
        return self.game.utility(state)

    # ---------------- The pipeline -------------------------------------- #
    def make_decision(self) -> Any | None:
        counter = 0
        while counter < self.n_iterations:

            # Step 1: Node selection
            node = self.select_ucb()
            if node is None:
                raise Exception(f"Ooops, no ucb selection from node\n{node}")
            if node.is_fully_expanded():
                raise Exception(f"Ooops, no expansion from node\n{node}")

            # Step 2: Expansion
            self.expand_node(node)

            # Step 3: Rollout
            rollout_result = self.make_rollout(node)

            # Step 4: Backpropagate
            self.backpropagate(node, rollout_result)

            counter += 1

        # Show best action
        best_root_action = self.get_best_root_action()  
        return best_root_action      

    # ---------------- heap helpers -------------------------------------- #
    def _push_node(self, node: SearchTreeNode, priority: float) -> None:
        heapq.heappush(self._frontier, (priority, self._counter, node))
        self._counter -= 1

    def _pop_best_node(self) -> SearchTreeNode:
        _, _, node = heapq.heappop(self._frontier)
        return node
    
    # ---------------- visualization helpers -------------------------------- #   
    def to_pydot(self) -> pydot.Dot:
        """Return a pydot graph representing the current search tree."""

        # ------------------------------------------------------------------ #
        # 1.  User-facing label function (make it a real Callable for mypy)
        # ------------------------------------------------------------------ #
        def label_fn(node: SearchTreeNode) -> str:
            player = self.game.player(node.state)
            msg = f"Value={node.value}\n"
            msg += f"To play={player}\n"
            msg += f"UCB={node.ucb(self.total_playouts):.2f}\n"
            msg += f"Finished={node.is_fully_expanded()}"
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