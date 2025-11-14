import json
import multiprocessing as mp
import os
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.erm_backward_induction import ERMBackwardInduction
from envs.envs import get_env, Occupancy_MDP

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

CONFIG = {
    "N": 1, # Number of experiments to run.
    "num_processors": 1,
    "env": "linear_mdp_fabio",
    "H": 5, # Truncation length.
    "n_iter_per_timestep": 1_000, # MCTS number of tree expansion steps per timestep.
    "erm_beta": 0.1,
}

def create_exp_name(args: dict) -> str:
    return args['env'] + '_' + args['algo'] + '_gamma_' + str(args['gamma']) + '_beta_' + str(args['erm_beta']) + '_' + \
        str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def simulate_ERM_backward_induction(mdp, H, erm_beta, n_iter_per_timestep=1_000):

    # Run ERMBackwardInduction algorithm.
    erm_algo = ERMBackwardInduction(mdp, mdp["gamma"], erm_beta, H)
    erm_opt_policy = erm_algo.compute()

    # Instantiate extended MDP.
    occupancy_mdp = Occupancy_MDP(mdp, H)

    # Sample initial state from the extended MDP.
    extended_state = occupancy_mdp.sample_initial_state()

    # Simulate until termination.
    cumulative_cost = 0.0
    for t in tqdm(range(H)):

        selected_action = erm_opt_policy[t, extended_state["state"]]

        # Environment step.
        extended_state, cost, terminated = occupancy_mdp.step(extended_state, selected_action)
        cumulative_cost += cost

    print("final cost:", cumulative_cost)

    return cumulative_cost


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate MDP.
    env = get_env(cfg["env"])
    print("env", env)

    mcts_f_val = simulate_ERM_backward_induction(mdp=env,
                               H=cfg["H"],
                               erm_beta=cfg["erm_beta"],
                               n_iter_per_timestep=cfg["n_iter_per_timestep"])

    return mcts_f_val


def main(cfg):

    if cfg["env"] not in ["linear_mdp", "linear_mdp_fabio"]:
        raise ValueError("ERM backward induction only works for the linear MDP environment.")

    # Setup experiment data folder.
    env = get_env(cfg["env"])
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "erm-backward-induction",
                                'gamma': env['gamma'],
                                'erm_beta': cfg['erm_beta'],
                                })
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('Config:')
    print(cfg)

    # Simulate.
    print('\nSimulating...')

    with mp.Pool(processes=cfg["num_processors"]) as pool:
        f_vals = pool.starmap(run, [(cfg, t) for t in range(cfg["N"])])
        pool.close()
        pool.join()

    f_vals = np.array(f_vals)

    exp_data = {}
    exp_data["config"] = cfg
    exp_data["f_vals"] = f_vals
    exp_data["env"] = env
    exp_data["env"]["f"] = None

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print(exp_name)

    return exp_name


if __name__ == "__main__":
    main(cfg = CONFIG)
