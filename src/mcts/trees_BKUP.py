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

from pprint import pprint

from itertools import count

from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Set

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
        action: Any,
        value: NodeValue,
        ucb_constant: float,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.value = value
        self.ucb_constant = ucb_constant

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

        if self.depth() == 1:  # parent is the root
            return self.action, self.value
        else:
            # recurse until we reach depth 1
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

    def __str__(self) -> str:
        root_flag = "--root--" if self.parent is None else ""
        s = f"State {root_flag}\n{self.state}\nDepth: {self.depth()}\n"
        s += f"Reward/Visits: {self.value}\n"
        if self.parent is not None:
            s += f"From action: {self.action}\nValue: {self.value}\n"
        return s


# --------------------------------------------------------------------------- #
#                       3.  Beam-width limited MCTS                           #
# --------------------------------------------------------------------------- #
class GameSearchTree:
    """
    Beam-width-limited Monte-Carlo Tree Search with an *ascending* heap frontier.
    """

    def __init__(
        self,
        state: Any,
        game: Any,
        rollout_policy: Callable[[Any], Any],
        sim_limit: int,
        beam_width: int,
        ucb_constant: float,
        seed: int
    ) -> None:
        # Random number generator
        self.rng = np.random.default_rng(42)

        # Hyper-parameters & helpers
        self.game = game
        self.rollout_policy = rollout_policy
        self.sim_limit = sim_limit
        self.beam_width = beam_width + 1
        self.ucb_constant = ucb_constant

        # Root node
        self.root = SearchTreeNode(
            state=state,
            parent=None,
            action=None,
            value=NodeValue(0, 0),
            ucb_constant=ucb_constant,
        )

        # Frontier implemented with `heapq`
        self._frontier: List[Tuple[float, int, SearchTreeNode]] = []
        self._counter = 0  # tie-breaker for equal priorities

        # Global playout counter
        self.total_playouts: int = 0

        # First expansion populates frontier and per-move stats
        self._root_actions: List[Any]
        self.action_stats = self._expand_root()


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

    # ---------------- heap helpers -------------------------------------- #
    def _push_leaf(self, node: SearchTreeNode, priority: float) -> None:
        heapq.heappush(self._frontier, (priority, self._counter, node))
        self._counter += 1

    def _pop_best_leaf(self) -> SearchTreeNode:
        _, _, node = heapq.heappop(self._frontier)
        return node

    # ---------------- tree expansion ------------------------------------ #
    def expand_node(self, node: SearchTreeNode) -> SearchTreeNode | None:
        actions = self.game.actions(node.state)
        if not actions:
            return
        if len(actions) > self.beam_width:
            actions = self.rng.choice(actions, self.beam_width, replace=False)

        children: List[SearchTreeNode] = []
        for a in actions:
            s_black = self.game.result(node.state, a)

            # Immediate terminal?
            if self.game.is_terminal(s_black):
                reward = 1 if self.game.utility(s_black) > 0 else 0
                self._backpropagate(node, reward)
                continue

            a_black = self.rollout_policy(s_black)
            s_white = self.game.result(s_black, a_black)

            child = SearchTreeNode(
                s_white, node, a, NodeValue(0, 0), self.ucb_constant
            )
            children.append(child)

        if not children:
            return None

        for c in children:
            self._push_leaf(c, c.ucb(self.total_playouts))

        
        
        return self.rng.choice(children)

    def _expand_root(self) -> Dict[str, NodeValue]:
        node = self.root
        actions = self.game.actions(node.state)
        if not actions:
            return {}

        if len(actions) > self.beam_width:
            actions = self.rng.choice(actions, self.beam_width, replace=False)
        self._root_actions = actions

        stats: Dict[str, NodeValue] = {str(a): NodeValue(0, 0) for a in actions}
        children: List[SearchTreeNode] = []

        for a in actions:
            s_white = self.game.result(node.state, a)

            if self.game.is_terminal(s_white):
                stats[str(a)] = NodeValue(1, 1)
                continue

            a_black = self.rollout_policy(s_white)
            s_black = self.game.result(s_white, a_black)

            child = SearchTreeNode(
                s_black, node, a, NodeValue(0, 0), self.ucb_constant
            )
            children.append(child)

        for c in children:
            self._push_leaf(c, c.ucb(self.total_playouts))
        return stats

    # ---------------- selection & backup -------------------------------- #
    def select_ucb(self) -> SearchTreeNode:
        """Pick the frontier node with **highest UCB** (lowest priority)."""
        return self._pop_best_leaf()

    def _backpropagate(self, node: SearchTreeNode, result: int) -> None:
        """Update stats and rebuild frontier priorities after each playout."""
        self.total_playouts += 1
        action, new_val = node.backpropagate(result)
        self.action_stats[str(action)] = new_val

        # Re-heapify every leaf with updated total_playouts
        new_heap: List[Tuple[float, int, SearchTreeNode]] = []
        self._counter = 0
        for _, _, leaf in self._frontier:
            heapq.heappush(
                new_heap, (leaf.ucb(self.total_playouts), self._counter, leaf)
            )
            self._counter += 1
        self._frontier = new_heap

    # ---------------- Policy rollout -------------------------------- #
    def make_rollout(self, node: SearchTreeNode, result: int) -> float:
        """Self-play using rollout policy"""
        counter = 0
        state = node.state.copy()
        while (counter < self.sim_limit) and (not self.game.is_terminal(state)):
            action = self.rollout_policy(state)
            state = self.game.result(state, action)
            counter += 1
        return self.game.utility(state)

    # ---------------- visualization helpers -------------------------------- #
    def build_children_dict(self) -> List[SearchTreeNode]:
        """Return {parent: [child, …]} for all nodes reachable from the leaves."""
        children = defaultdict(list)
        leafs = [t[-1] for t in self._frontier]
        for leaf in leafs:
            node = leaf
            while node.parent is not None:
                children[node.parent].append(node)
                node = node.parent
        return children
    
    def to_pydot(self) -> pydot.Dot:
        """Return a pydot graph representing the current search tree."""

        # ------------------------------------------------------------------ #
        # 1.  User-facing label function (make it a real Callable for mypy)
        # ------------------------------------------------------------------ #
        label_fn: Callable[[SearchTreeNode], str] = (
            lambda n: "root" if n.action is None else f"{n.action}\nValue={n.value}"
        )

        # ------------------------------------------------------------------ #
        # 2.  Build a *deduplicated* children map
        # ------------------------------------------------------------------ #
        raw_children: Dict[SearchTreeNode, List[SearchTreeNode]] = self.build_children_dict()
        # convert list → set to kill duplicates in O(N)
        children: Dict[SearchTreeNode, Set[SearchTreeNode]] = {
            parent: set(child_list) for parent, child_list in raw_children.items()
        }

        # ------------------------------------------------------------------ #
        # 3.  Initialise the pydot graph
        # ------------------------------------------------------------------ #
        graph: pydot.Dot = pydot.Dot(graph_type="digraph", rankdir="TB")

        # Unique string id for each SearchTreeNode
        _id_counter = count()
        id_map: Dict[SearchTreeNode, str] = {}

        def get_id(node: SearchTreeNode) -> str:
            """Return a stable string id for *node* (create on first call)."""
            if node not in id_map:
                id_map[node] = f"n{next(_id_counter)}"
            return id_map[node]

        # ------------------------------------------------------------------ #
        # 4.  DFS while tracking visited nodes to avoid re-adding edges/nodes
        # ------------------------------------------------------------------ #
        visited: Set[SearchTreeNode] = set()
        stack: List[SearchTreeNode] = [self.root]

        while stack:
            node: SearchTreeNode = stack.pop()

            if node in visited:          # already processed → skip
                continue
            visited.add(node)

            node_id: str = get_id(node)
            graph.add_node(
                pydot.Node(node_id, label=label_fn(node), shape="box")
            )

            for child in children.get(node, set()):
                child_id: str = get_id(child)
                graph.add_edge(pydot.Edge(node_id, child_id))
                stack.append(child)

        return graph