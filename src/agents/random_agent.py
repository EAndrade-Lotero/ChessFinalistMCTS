from agents.base_classes import PolicyAgent

class RandomAgent(PolicyAgent):
<<<<<<< HEAD

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
=======
    def __init__(
        self,
        policy,
        n_actions,
        action_encoder: int,
        rng=None,
        debug=False
    ) -> None:
        super().__init__(
            policy=policy,
            n_actions=n_actions,
            action_encoder=action_encoder,
            rng=rng,
>>>>>>> 7acc1d490a4d56556d40e337aff9817a7bf32c5e
            debug=debug
        )

    def update(self, next_state, reward, done):
        """Do nothing"""
        return
