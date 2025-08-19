from agents.base_classes import PolicyAgent


class RandomAgent(PolicyAgent):

    def __init__(self, *, policy, n_actions, rng = None, debug = False):
        super().__init__(policy=policy, n_actions=n_actions, rng=rng, debug=debug)

    def update(self, next_state, reward, done):
        """Do nothing"""
        return

    