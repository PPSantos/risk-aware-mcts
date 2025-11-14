from simulate_cvar_mcts import main as run

CONFIG = {
    "N": 100, # Number of experiments to run.
    "num_processors": 50,
    "env": "entropy_mdp",
    "H": 100, # Truncation length.
    "n_iter_per_timestep": 1_000, # MCTS number of tree expansion steps per timestep.
    # "cvar_alpha": 0.9,
}

exp_ids = []
for alpha in [0.05, 0.5, 0.95]:
    print("alpha=", alpha)
    CONFIG["cvar_alpha"] = alpha
    exp_id = run(CONFIG)
    exp_ids.append(exp_id)

print(exp_ids)
