from stable_baselines3 import A2C
from typing import Any

from agents.base_classes import GameProtocol
from agents.random_policy import GameUniformPolicy

class StBlsA2C(GameUniformPolicy):

    def __init__(self, game: GameProtocol, rng: Any | None = None) -> None:
        super().__init__(game, rng)
        