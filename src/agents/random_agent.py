from agents.base_classes import PolicyAgent

class RandomAgent(PolicyAgent):

    def __init__(
        self, 
        policy,
        rng = None,
        debug = False
    ) -> None:
        super().__init__(
            policy=policy,  
            rng=rng,
            n_actions = None,
            debug=debug
        )

    def update(self, next_state, reward, done):
        """Do nothing"""
        return

    def super_make_decision():
        pass