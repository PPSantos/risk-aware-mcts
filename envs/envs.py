import numpy as np


MDPs = {

    "four_state_mdp": {
        "states": [0, 1, 2, 3],
        "actions": [0, 1], # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0, 0.0],
        "P": np.array([
            [
            [0.0, 0.0, 0.85, 0.15],
            [0.1, 0.9, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.0, 0.0, 0.9],
            ],
            [
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.1, 0.0, 0.9, 0.0],
            [0.1, 0.0, 0.0, 0.9],
            ],
        ]),
        "C": np.array([[0.0, 0.0], # c=0 / 20.
                       [0.25, 0.25], # c=5 / 20.
                       [0.05, 0.05], # c=1 / 20.
                       [1.0, 1.0]]) # c=20 / 20.
    },

}

class Env:

    def __init__(self, mdp, H):

        self.mdp = mdp
        self.H = H
        self.gamma = mdp["gamma"]

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        # Sample initial state.
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        extended_state = {"state": state,
                          "t": 0} # (state, timestep).
        return extended_state
    
    def step(self, extended_state, a):

        # Simulate a step of the finite-horizon MDP.
        state_t, timestep_t = extended_state["state"], extended_state["t"]
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a,state_t,:])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state,
                               "t": next_timestep}
        
        cost_t = self.mdp["C"][state_t, a]

        if next_timestep >= self.H:
            terminated = True
        else:
            terminated = False

        return next_extended_state, cost_t, terminated


def get_env(env_name, H):
    env_dict = MDPs[env_name]
    return Env(env_dict, H)
