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

from tqdm.auto import tqdm

from pprint import pprint
from itertools import count
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, Set, Optional

from agents.base_classes import PolicyProtocol, GameProtocol


import heapq
import itertools
from typing import Generic, TypeVar, Tuple

T = TypeVar("T")

# --------------------------------------------------------------------------- #
#                           0.  Priority Queue                                #
# --------------------------------------------------------------------------- #
class PriorityQueue(Generic[T]):
    """
    Minimal priority queue backed by heapq.

    - Min-heap by default; pass max_heap=True for max-heap behavior.
    - Stable for equal priorities (insertion order preserved).
    """

    def __init__(self, *, max_heap: bool = True) -> None:
        self._heap: list[tuple[int | float, int, T]] = []
        self._counter = itertools.count()  # tie-breaker for stability
        self._sign = -1 if max_heap else 1

    def push(self, priority: int | float, item: T) -> None:
        """Insert (priority, item)."""
        heapq.heappush(self._heap, (self._sign * priority, next(self._counter), item))

    def pop(self) -> Tuple[int | float, T]:
        """
        Remove and return the (priority, item) with best priority.
        Raises IndexError if empty.
        """
        if not self._heap:
            raise IndexError("pop from empty PriorityQueue")
        p, _, item = heapq.heappop(self._heap)
        return (self._sign * p, item)

    # (Optional niceties)
    def __len__(self) -> int:  # allows: len(pq)
        return len(self._heap)

    def __bool__(self) -> bool:  # allows: if pq:
        return bool(self._heap)


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
        player: str,
        parent: "SearchTreeNode | None",
        action: "Any | None",
        untried_actions: List[Any],
        value: NodeValue,
        ucb_constant: float
    ) -> None:
        self.state = state
        self.player = player
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
        root: Any | SearchTreeNode,
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
                "`rollout_policy` must implement a Policy Protocol."
            )
        self.rollout_policy = rollout_policy

        self.sim_limit = sim_limit
        self.beam_width = beam_width
        self.ucb_constant = ucb_constant
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

        # Frontier implemented with `heapq`
        self._frontier: List[Tuple[float, int, SearchTreeNode]] = []
        self._counter = 0  # tie-breaker for equal priorities

    # ---------------- public helper to pick the best root move ----------- #
    def get_best_root_action(self) -> Any | None:
        """Pick the root's action with highest UCB"""
        best_child = self.get_child_with_highest_ucb(self.root, skip_terminals=False)
        if best_child is None:
            return None
        else:
            return best_child.action

    def get_child_with_highest_ucb(self, node: SearchTreeNode, skip_terminals: Optional[bool] = True) -> Any | None:
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

        child = None
        while child is None and children:
            best_ucb, child = children.pop()
            is_terminal = self.game.is_terminal(child.state)
            if is_terminal and skip_terminals: 
                child = None

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
            ucb_constant=self.ucb_constant
        )

        if self.game.is_terminal(child.state):
            result = self.game.utility(child.state)
            self.backpropagate(child, result)

        node.children.append(child)

    def _expand_root(self, state: Any) -> None:
        # Find Player
        player = self.game.player(state)

        # Get root actions
        actions = self.game.actions(state)
        if not actions:
            raise Exception('Error: No valid actions from root!')

        if len(actions) > self.beam_width:
            actions = self.rng.choice(actions, self.beam_width, replace=False).tolist()
        self._root_actions = actions

        # Root node
        self.root = SearchTreeNode(
            state=state,
            player=player,
            parent=None,
            action=None,
            untried_actions=[],
            value=NodeValue(0, 0),
            ucb_constant=self.ucb_constant,
        )

        for a in self._root_actions:
            new_state = self.game.result(state, a)
            # Find Player
            player = self.game.player(new_state)

            untried_actions = self.game.actions(new_state)
            if len(untried_actions) > self.beam_width:
                untried_actions = self.rng.choice(untried_actions, self.beam_width, replace=False).tolist()

            child = SearchTreeNode(
                state=new_state,
                player=player,
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
        """Tree search strategy"""
        node = self.root
        best_node = self._recursive_select_ucb(node)
        return best_node

    def _recursive_select_ucb(self, node: SearchTreeNode) -> SearchTreeNode | None:
        """Recursive search"""

        best_child = self.get_child_with_highest_ucb(node)
        if best_child is None:
            print(f"{node.state}")
            raise Exception(f"Ooops, no ucb selection from node\n{node}")

        if not best_child.is_fully_expanded():
            return best_child
        else:
            return self._recursive_select_ucb(best_child)


        # multiplier = 1 if self.game.player(node.state) == 'white' else -1

        # for child in node.children:
        #     if not self.game.is_terminal(child.state):
        #         ucb = child.ucb(self.total_playouts) * multiplier
        #         self._push_node(child, ucb)

        # while len(self._frontier) > 0:
        #     best_child = self._pop_best_node()
        #     # print(f"Checking {best_child.action}...")
        #     if not best_child.is_fully_expanded():
        #         # print("This is the node!")
        #         return best_child
        #     else:
        #         # print("Entering recursion")
        #         return self._recursive_select_ucb(best_child)
        return None

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
        counter = 0
        while counter < self.n_iterations:

            # Step 1: Node selection
            node = self.get_child_with_highest_ucb(self.root)
            if node is None:
                raise Exception(f"Ooops, no ucb selection from node\n{node}")
            elif node.is_fully_expanded():
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
        heapq.heappush(self._frontier, (-priority, self._counter, node))
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
            msg += f"Finished={node.is_fully_expanded()}\n"
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