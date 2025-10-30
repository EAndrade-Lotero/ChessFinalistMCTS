from agents.base_classes import PolicyAgent


class RandomAgent(PolicyAgent):

    def __init__(
        self, 
        policy, 
        n_actions, 
        action_encoder: int,        
        rng = None, 
        debug = False
    ) -> None:
        super().__init__(
            policy=policy, 
            n_actions=n_actions,
            action_encoder=action_encoder, 
            rng=rng, 
            debug=debug
        )

    def update(self, next_state, reward, done):
        """Do nothing"""
        return

    