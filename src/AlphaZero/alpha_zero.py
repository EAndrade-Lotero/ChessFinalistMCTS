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

import pydot
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy.random import Generator, default_rng
from typing import Generic, TypeVar, Tuple

from pprint import pprint
from itertools import count
from typing import Any, Callable, Dict, List, Tuple, Set, Optional
from stable_baselines3.a2c import A2C

from agents.base_classes import PolicyProtocol, GameProtocol, EncoderProtocol
from mcts.trees import PriorityQueue, NodeValue
from games.chess import KRK
from agents.utils import ChessEncoder
from agents.random_agent import RandomAgent
from agents.random_policy import GameUniformPolicy
from games.game_utils import GymEnvFromGameAndPlayer2

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

        self.value.reward += result

        # if self.player == "white":
        #     #If Player is white, black has just played
        #     if result == 1:
        #         #Black Lost
        #         pass
        #     elif result == -1:
        #         #Black Won
        #         self.value.reward -= result
        # elif self.player == "black":
        #     #If Player is black, white has just played
        #     if result == 1:
        #         #White Won
        #         self.value.reward += result
        #     elif result == -1:
        #         #White Lost
        #         pass

        self.value.playouts += 1

        if self.depth() == 0:  # parent is the root
            return self.action, self.value
        else:
            # recurse until we reach depth 0
            return self.parent.backpropagate(result)  # type: ignore[arg-type]

    def puct(self, total_visits: int, p: float) -> float:
        """PUCT"""
        exploration = self.puct_constant * p * np.sqrt(total_visits) / (1 + self.value.playouts)
        return self.mean_value() + exploration

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
        # if self.finished:
        #     return True
        # return np.all([child.finished for child in self.children])
        return self.finished

    def __str__(self) -> str:
        root_flag = "--root--" if self.parent is None else ""
        s = f"State {root_flag}\n{self.state}\nDepth: {self.depth()}\n"
        s += f"Reward/Visits: {self.value}\n"
        if self.parent is not None:
            s += f"From action: {self.action}\nValue: {self.value}\n"
        return s

    def is_fully_expanded(self) -> bool:
        return len(self.children) != 0

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
        value = self.get_v(self.root)
        
        # Create priority queue with children
        children = PriorityQueue()
        for child in self.root.children:
            p = self.get_p(child)
            puct = child.puct(self.total_playouts, p)
            children.push(puct, child)

        _, best_child = children.pop()
        if best_child is None:
            return None
        else:
            return best_child.action

    def get_child_with_highest_puct(
        self, 
        node: SearchTreeNode, 
        skip_finished: Optional[bool] = True,
        skip_terminals: Optional[bool] = True,
    ) -> Any | None:
        """Pick the node's child with highest PUCT"""
        best_puct = -np.inf
        matches = []

        # Create priority queue with children
        children = PriorityQueue()
        for child in node.children:
            p = self.get_p(child)
            puct = child.puct(self.total_playouts, p)
            children.push(puct, child)

        # print(f"Queue: {children}")

        # Iterate to find child with best ucb with conditions
        child = None
        while child is None and children:
            best_puct, child = children.pop()
            is_terminal = self.game.is_terminal(child.state)
            # print(
            #     f"Child {child.action} " 
            #     f"with PUCT {best_puct:.2f} " 
            #     f"and terminal={is_terminal} " 
            #     f"and finished={child.is_finished()}"
            # )
            # if is_terminal and skip_terminals: 
            #     child = None
            #     continue
            if child.is_finished() and skip_finished:
                child = None
                continue

        # print(
        #     f"Child {child.action} " 
        #     f"with PUCT {best_puct:.2f} " 
        #     f"and terminal={is_terminal} " 
        #     f"and finished={child.is_finished()}"
        # )

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

        state = node.state

        # Get root actions
        actions = self.game.actions(state)
        if not actions:
            raise Exception('Error: No valid actions from root!')

        for a in actions:
            new_state = self.game.result(state, a)
            # Find Player
            player = self.game.player(new_state)

            child = SearchTreeNode(
                state=new_state,
                player=player,
                parent=node, 
                action=a, 
                value=NodeValue(0, 0),
                puct_constant=self.puct_constant,
            )

            if self.game.is_terminal(child.state):
                # print(f"Terminal child from root with action {a} and state\n{child.state}")
                child.finished = True
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
                puct_constant=self.puct_constant,
            )

            if self.game.is_terminal(child.state):
                # print(f"Terminal child from root with action {a} and state\n{child.state}")
                child.finished = True
                result = self.game.utility(child.state)
                self.backpropagate(child, result)

            self.root.children.append(child)
    
    # ---------------- selection & backup -------------------------------- #
    def select_puct(self) -> SearchTreeNode | None:
        """Tree search strategy"""
        node = self.root
        best_node = self._recursive_select_puct(node)
        return best_node

    def _recursive_select_puct(self, node: SearchTreeNode) -> SearchTreeNode | None:
        """Recursive search"""

        best_child = self.get_child_with_highest_puct(
            node=node, 
            skip_finished=True
        )

        # print(
        #     f"Child {best_child.action} " 
        #     f"and terminal={best_child.finished} " 
        #     f"and finished={best_child.is_finished()}"
        # )

        if best_child is None:
            if len(node.children) == 0 and not node.is_finished():
                return node
            print(
                f"{node.state}"
                f"\nPuct: {node.puct(self.total_playouts, self.get_p(node))}"
                f"\nChildren: {[n.action for n in node.children]}"
                f"\nfinished={node.is_finished()}"
            )
            raise Exception(f"Ooops, didn't find usable best child from node\n{node}\nThe action history is:\n{node.get_action_history()}")

        # print(
        #     f"Selected child {best_child.action} is fully expanded: {best_child.is_fully_expanded()} " 
        #     f"and finished: {best_child.is_finished()} "
        #     f"Num. children: {len(best_child.children)}"
        # )
        if not best_child.is_fully_expanded():
            # len == 0
            return best_child
        
        elif best_child.is_finished():
            return None
        
        
        return self._recursive_select_puct(best_child)

    def backpropagate(self, node: SearchTreeNode, result: int) -> None:
        """Update stats and rebuild frontier priorities after each playout."""
        self.total_playouts += 1
        action, new_val = node.backpropagate(result)

    # ---------------- The pipeline -------------------------------------- #
    def make_decision(self) -> Any | None:

        if self.root.is_finished():
            return self.get_best_root_action()
        
        counter = 0
        while counter < self.n_iterations:

            # Step 1: Node selection
            node = self.select_puct()
            if node is None:
                raise Exception(f"Ooops, no puct selection from node\n{node}\n{node.get_action_history()}")
            elif node.is_fully_expanded():
                raise Exception(f"Ooops, no expansion from node\n{node}\n{node.get_action_history()}")
            elif node.is_finished():
                raise Exception(f"Ooops, finished node not for expansion\n{node}\n{node.get_action_history()}")

            # Step 2: Expansion
            self.expand_node(node)

            # Step 3: Rollout
            result = self.get_v(node)

            # Step 4: Backpropagate
            self.backpropagate(node, result)

            counter += 1

        # Get best action
        return self.get_best_root_action()  
  
    def to_tensor(self, node: SearchTreeNode, device: str='cpu') -> torch.tensor:
        state_ = self.encoder.encode_obs(node.state)
        state_ = self.encoder.to_array(state_)
        s_tensor = torch.tensor(state_, dtype=torch.float32).to(device)
        # s_tensor /= 3
        return s_tensor

    def get_p(self, node: SearchTreeNode) -> float:
        state_ = self.encoder.encode_obs(node.state)
        state_ = self.encoder.to_array(state_)
        action = self.encoder.encode_action(node.parent.state, node.action)
        p = self.policy_network(state_).squeeze().tolist()[action]
        return p

    def get_v(self, node: SearchTreeNode) -> float:
        state_ = self.encoder.encode_obs(node.state)
        state_ = self.encoder.to_array(state_)
        v = self.value_network(state_).squeeze().item()
        return v

    # ---------------- visualization helpers -------------------------------- #   
    def to_pydot(self) -> pydot.Dot:
        """Return a pydot graph representing the current search tree."""

        # ------------------------------------------------------------------ #
        # 1.  User-facing label function (make it a real Callable for mypy)
        # ------------------------------------------------------------------ #
        def label_fn(node: SearchTreeNode) -> str:
            p = None
            state = node.state
            # print("="*60)
            # print(state)
            # print(f"Move: {node.action}")
            # print("="*60)

            player = self.game.player(state)

            if node.action is not None:
                # v = self.value_network(state_).item()
                # state_ = self.encoder.encode_obs(state)
                # state_ = self.encoder.to_array(state_)
                # action = self.encoder.encode_action(node.parent.state, node.action)
                # # print(f"====>{action=}")
                # p = self.policy_network(state_).squeeze().tolist()[action]
                p = self.get_p(node)
                puct = node.puct(self.total_playouts, p)
                # print(puct, type(puct))
            else:
                puct = 0.0

            node_value = self.get_v(node)

            msg = f"Network_Value={node_value:.2f}\n"
            if p is not None:
                msg += f"Network_prob={p:.2f}\n"
            msg += f"MCT_Value={node.value}\n"
            msg += f"To play={player}\n"
            msg += f"PUCT={puct:.2f}\n"
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
    
    def padded_policy(self, node: SearchTreeNode) -> np.ndarray:

        def get_frequency_for_action(node, move, sum):
            assert isinstance(move, chess.Move)
            for child in node.children:
                if child.action == move:
                    return child.value.playouts / sum
            return 0.0

        policy = np.array([
            children.value.playouts for children in node.children
        ])

        sum_ = sum(policy)
        policy_every_action = np.zeros(self.encoder.n_actions)

        for i in range(self.encoder.n_actions):
            try:
                move = self.encoder.decode_action(state, i)
                policy_every_action[i] = get_frequency_for_action(self.root, move, sum_)
            except Exception as e:
                # print(f"Error decoding action index {i}: {e}")
                policy_every_action[i] = 0.0

        return policy_every_action
        ######
        # Poner 0 en las acciones que no se pueden

        # Recorrer las 30 posibles acciones
        # Con try -> mantiene el valor
        # Con except -> pone 0
        ######
        # pass


class AlphaZeroDataset(Dataset):

    def __init__(self, tree: GameSearchTree, max_size: int = 100):
        # Cambiamos la lista por un deque con tamaño máximo
        self.samples = deque(maxlen=max_size)
        self.tree = tree
        self.device = 'cpu'

    def add(self, node: SearchTreeNode) -> None:
        # state
        s_tensor = self.tree.to_tensor(node)

        # policy
        policy = self.tree.padded_policy(node)

        # root_value
        root_value = node.mean_value()

        self.add_sample(s_tensor, policy, root_value)

    def add_sample(self, state, policy, root_value):
        """
        state  -> representación del tablero
        policy -> distribución MCTS
        root_value  -> resultado final z
        """
        self.samples.append(
            (
                torch.tensor(state, dtype=torch.float32).to(self.device),
                torch.tensor(policy, dtype=torch.float32).to(self.device),
                torch.tensor(root_value, dtype=torch.float32).to(self.device),
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def create_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class SelfPlay:

    hyperparameters = {
        "start_position_range": [1, 3],
        "batch_size": 4,
        "n_epochs": 2,
        "n_timesteps": 10,
        "device": "cpu",
        "train_every": 5,
        "min_samples_to_train": 5,
        "buffer_size": 20,
    }

    def __init__(self, start_position, puct_constant):
        self.seed = 4
        self.rng = default_rng(self.seed)
        
        self.game = KRK(start_position=start_position)
        random_policy = GameUniformPolicy(
            game=self.game,
            rng=self.rng,
            encoder=None
        )
        pl2 = RandomAgent(
            policy=random_policy,
            rng=self.rng
        )
        encoder = ChessEncoder()
        self.env = GymEnvFromGameAndPlayer2(
            game=self.game,
            other_player=pl2,
            encoder=encoder
        )
        
        self.agent = A2C('MlpPolicy', self.env, verbose=0)
        self.rewards = []

        self.params = {
            "puct_constant": puct_constant,
            "value_network": self.value_network,
            "policy_network": self.policy_network,
            "encoder": encoder,
            "n_iterations": 100,
            "rng":self.rng
        }
        state = self.game.initial_state
        self.tree = GameSearchTree(
            root=state,
            game=self.game,
            **self.params
        )

        self.dataset = AlphaZeroDataset(self.tree)
        self.dataloader = None        
        self.debug = True

    def value_network(self, state):
        device = self.hyperparameters["device"]
        # Convert state to tensor
        tensor = torch.tensor(state, dtype=torch.float32).to(device)
        # Add batch dimension
        tensor = tensor.unsqueeze(dim=0)
        # Get value prediction from the policy
        with torch.no_grad():
            value = self.agent.policy.predict_values(tensor)
        return value

    def policy_network(self, state):
        device = self.hyperparameters["device"]
        # Convert state to tensor
        tensor = torch.tensor(state, dtype=torch.float32).to(device)
        # Add batch dimension
        tensor = tensor.unsqueeze(dim=0)
        # Get value prediction from the policy
        with torch.no_grad():
            distribution = self.agent.policy.get_distribution(tensor)
            probs = distribution.distribution.probs
        return probs

    def step(self):

        best_action = self.tree.make_decision()

        if self.debug:
            print(f"We are here:\n{self.tree.root.state}")
            print(f"Best action: {best_action}")

        # state = self.game.result(self.tree.root.state, best_action)
        self.dataset.add(self.tree.root)
        root = self.tree.get_root_child_from_action(best_action)
        self.tree =GameSearchTree(
            root=root,
            game=self.game,
            **self.params,
        )

    def train_epoch(self):
        if self.dataloader is not None:
            for states, target_pi, target_z in self.dataloader:

                # ---------------------------------
                # Forward pass through shared encoder
                # ---------------------------------

                features = self.agent.policy.extract_features(states)

                latent_pi, latent_vf = self.agent.policy.mlp_extractor(features)

                # ---------------------------------
                # POLICY HEAD
                # ---------------------------------

                pred_pi_logits = self.agent.policy.action_net(latent_pi)
                # print(f"pred_pi_logits: {pred_pi_logits}")

                # ---------------------------------
                # VALUE HEAD
                # ---------------------------------

                pred_v = self.agent.policy.value_net(latent_vf)

                # ---------------------------------
                # VALUE LOSS
                # ---------------------------------

                value_loss = torch.mean(
                    (target_z - pred_v.squeeze()) ** 2
                )

                # ---------------------------------
                # POLICY LOSS
                # ---------------------------------

                log_probs = torch.log_softmax(
                    pred_pi_logits,
                    dim=1,
                )
                print(f"log_probs: {log_probs.shape}")
                print(f"target_pi: {target_pi.shape}")

                policy_loss = -torch.mean(
                    torch.sum(
                        target_pi * log_probs,
                        dim=1,
                    )
                )

                # ---------------------------------
                # L2 REGULARIZATION
                # ---------------------------------

                l2_lambda = 1e-4

                l2_loss = sum(
                    p.pow(2).sum()
                    for p in self.agent.policy.parameters()
                )

                # ---------------------------------
                # TOTAL LOSS
                # ---------------------------------

                loss = (
                    value_loss
                    + policy_loss
                    + l2_lambda * l2_loss
                )

                # ---------------------------------
                # OPTIMIZATION STEP
                # ---------------------------------

                self.agent.policy.optimizer.zero_grad()

                loss.backward()

                self.agent.policy.optimizer.step()

                if self.debug:
                    print(
                        f"value_loss={value_loss.item():.4f} | "
                        f"policy_loss={policy_loss.item():.4f} | "
                        f"total={loss.item():.4f}"
                    )        

    def train(self):
        n_epochs = self.hyperparameters["n_epochs"]
        for epoch in range(n_epochs):
            self.train_epoch()

    def run_timesteps(self):
        """
        Args:
            n_timesteps: Número total de pasos a ejecutar.
            train_every: Cada cuántos timesteps se ejecutará el entrenamiento.
            min_samples_to_train: Mínimo de datos requeridos en el dataset para entrenar.
        """
        self.rewards = []

        batch_size = self.hyperparameters["batch_size"]
        self.dataloader = self.dataset.create_dataloader(batch_size=batch_size, shuffle=True)

        n_timesteps = self.hyperparameters["n_timesteps"]
        train_every = self.hyperparameters["train_every"]
        min_samples_to_train = self.hyperparameters["min_samples_to_train"]

        for timestep in range(1, n_timesteps + 1):

            if self.game.is_terminal(self.tree.root.state):

                reward = self.game.utility(self.tree.root.state)
                self.rewards.append(reward)
                if self.debug:
                    print(f"Reward: {reward}")

                start_position = self.rng.choice(range(1, 3))
                self.reset_game(start_position)

                if self.debug:
                    print(f"Game reset to position {start_position}:\n{self.game.initial_state}")

            else:
                self.step()

            # Disparar el entrenamiento cada 'train_every' pasos
            if timestep % train_every == 0:
                if len(self.dataset) >= min_samples_to_train:
                    if self.debug:
                        print(f"[Timestep {timestep}] Iniciando fase de entrenamiento...")
                    self.train()
                # elif self.debug:
                #     print(f"[Timestep {timestep}] Entrenamiento saltado: datos insuficientes ({len(self.dataset)}/{min_samples_to_train})")
    
    def reset_game(self, start_position=1):
        assert self.hyperparameters["start_position_range"][0] <= start_position
        assert start_position <= self.hyperparameters["start_position_range"][1]
        self.game.reset(start_position=start_position)
        self.tree = GameSearchTree(
            root=self.game.initial_state,
            game=self.game,
            **self.params,
        )

    ##### Falta:
    # - Usar self.rewards para guardar el utility de cada estado terminal
    # - Reportar una grafica de rewards a lo largo del tiempo