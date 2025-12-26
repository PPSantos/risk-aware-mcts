from simulate_mcts_accrued_costs import main as run

CONFIG = {
    "N": 100, # Number of experiments to run.
    "num_processors": 50,
    "env": "four_state_mdp",
    "H": 100, # Truncation length.
    "n_iter_per_timestep": 100, # MCTS number of tree expansion steps per timestep.
    # "erm_beta": 0.1,
}

exp_ids = []
for erm_beta in [0.1, 0.5, 1.0]:
    print("erm_beta=", erm_beta)
    CONFIG["erm_beta"] = erm_beta
    exp_id = run(CONFIG)
    exp_ids.append(exp_id)

print(exp_ids)
