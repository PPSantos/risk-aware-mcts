import numpy as np
# import matplotlib.pyplot as plt

# Cliff MDP situation
ncols=10
nrows=10
delta=0.01
obstacle_prob_max=0.4

rng = np.random.default_rng(seed=6)

g = 0.99

# State i corresponds to coordinates (r, c):
# i = r * ncols + c
n_states = ncols * nrows
states = list(range(n_states))

def to_state(r, c):
    return r * ncols + c

def to_rc(s):
    return divmod(s, ncols)

start = to_state(0, ncols - 1)
goal  = to_state(nrows - 1, ncols - 1)

# 0 = up, 1 = down, 2 = left, 3 = right
actions = [0, 1, 2, 3]
n_actions = len(actions)

# Movements for actions
moves = {
    0: (-1, 0),   # up
    1: (1,  0),   # down
    2: (0, -1),   # left
    3: (0,  1),   # right
}

# Probability decreases linearly from right to left
obstacle_prob = np.linspace(0.0, obstacle_prob_max, ncols)

obstacles = np.zeros((nrows, ncols), dtype=bool)
for r in range(nrows):
    for c in range(ncols):
        if (r, c) in [(0, ncols - 1), (nrows - 1, ncols - 1)]:
            continue  # start and goal cannot be obstacles
        if r in [0, 1, nrows-2, nrows - 1]:
            continue # I don't want this
        if rng.random() < obstacle_prob[c]:
            obstacles[r, c] = True


'''obstacles = np.array([[False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False,  True],
 [False, False,  True, False, False, False,  True,  True, False, False],
 [False, False, False, False, False, False,  True, False,  True, False],
 [False, False, False, False, False, False, False,  True, False,  True],
 [False, False,  True, False, False, False,  True,  True, False, False],
 [False, False, False, False,  True, False, False,  True,  True, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False]])'''

'''obstacles = np.array([
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, True, False, False, False, True, False, True],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False]])'''


'''obstacles = np.array([
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, True, False, False, False, False, False, True],
 [False, False, False, False, False, False, False, True, False, False],
 [False, False, False, True, False, False, False, False, False, True],
 [False, False, False, False, False, False, False, True, False, False],
 [False, False, False, True, False, False, False, False, False, True],
 [False, False, False, False, False, False, False, True, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False]])'''


obstacles = np.array([
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, True, False, False, False, True, True, False],
 [False, False, False, True, False, False, False, False, True, False],
 [False, False, False, False, False, False, True, False, False, False],
 [False, False, False, False, False, False, False, True, False, False],
 [False, False, False, True, False, False, False, False, True, False],
 [False, False, False, False, False, False, False, True, True, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False]])


'''obstacles = np.array([
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, True],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, True, False, False, False],
 [False, False, False, True, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, True],
 [False, False, False, False, False, False, False, False, False, False],
 [False, False, False, False, False, False, False, False, False, False]])
'''

# Costs
C = np.ones((n_states, n_actions)) # default cost = 1
C[goal, :] = 0.0  # goal is absorbing with zero cost
collision_cost = 275 # 2 / (1-g)

# Transitions
P = np.zeros((n_actions, n_states, n_states))

def attempt_move(r, c, a):
    dr, dc = moves[a]
    nr, nc = r + dr, c + dc

    # If we step outside the grid
    if not (0 <= nr < nrows and 0 <= nc < ncols):
        return r, c  # stay in place
        #return  nrows-4, ncols-1 # going out of grid takes to absorbing state

    return nr, nc

# ---- Build transitions ----
for s in states:
    if s == goal:
        # Absorbing state
        P[:, s, s] = 1.0
        continue

    r, c = to_rc(s)
    if obstacles[r, c]:
        C[s, :] = collision_cost
    '''if obstacles[r, c]:
        # Absorbing state
        P[:, s, s] = 1.0
        C[s,:] = collision_cost
        continue'''

    for a in actions:
        # With prob 1 - delta take chosen action
        nr, nc = attempt_move(r, c, a)
        s_prime = to_state(nr, nc)

        prob_correct = 1 - delta
        P[a, s, s_prime] += prob_correct

        # With prob delta uniform random direction
        random_prob = delta / n_actions
        for ar in actions:
            nr2, nc2 = attempt_move(r, c, ar)
            sp2 = to_state(nr2, nc2)
            P[a, s, sp2] += random_prob


# Normalize numerically (safety)
for a in actions:
    for s in states:
        P[a, s, :] /= P[a, s, :].sum()

p0 = np.zeros(n_states)
p0[start] = 1.0

'''mdp = {
    "states": states,
    "actions": actions,# 0 = up, 1 = down, 2 = left, 3 = right
    "gamma": g,
    "p_0": p0,
    "P": P,
    "C": C,
    "grid_width": ncols,
    "grid_height": nrows,
    "obstacles": obstacles,
    "start": start,
    "goal": goal,
}'''



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
    "obstacle_mdp": {
        "states": states,
        "actions": actions,
        "gamma": 0.9,
        "p_0": p0,
        "P": P,
        "C": C
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


if __name__ == "__main__":
    env = get_env("obstacle_mdp", 100)
    print(obstacles)
    color_map = np.zeros(obstacles.shape + (3,), dtype=float)  # Create a color array, shape (height, width, 3)

    # Set red for True (obstacles) and white for False (no obstacles)
    color_map[obstacles] = [1, 0, 0]  # Red for obstacles (R, G, B)
    color_map[~obstacles] = [0, 0, 1]  # White for non-obstacles

    plt.figure(figsize=(6, 6))
    plt.imshow(color_map, interpolation='nearest')
    plt.axis('off')  # Hide the axes
    plt.title("Obstacle Grid")
    plt.show()
    # Plotting
    '''for a in actions:
        print("Cost dest", C[goal,a])
        data = C[:, a].reshape((nrows, ncols))
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap='hot',
                   interpolation='nearest')  # You can change 'hot' to other colormaps like 'viridis', 'plasma', etc.
        plt.colorbar(label='Value Scale')  # Show color scale
        plt.title("Heatmap of the 2D Float Matrix")
        plt.axis('off')  # Hide the axes
        plt.show()'''
