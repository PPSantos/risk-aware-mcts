import json
import multiprocessing as mp
import os
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.erm_backward_induction import ERMBackwardInduction
from envs.envs import get_env, MDPs

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)
DEBUG = False

CONFIG = {
    "N": 100, # Number of experiments to run.
    "num_processors": 10,
    "env": "four_state_mdp",
    "H": 100, # Truncation length.
    "erm_beta": 1.0,
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

def simulate_ERM_backward_induction(env, H, erm_beta):

    # Run ERMBackwardInduction algorithm.
    erm_algo = ERMBackwardInduction(env, erm_beta, H)
    erm_opt_policy = erm_algo.compute()

    # Sample initial state.
    extended_state = env.sample_initial_state()

    # Simulate until termination.
    cumulative_discounted_cost = 0.0
    for t in tqdm(range(H)):

        if DEBUG:
            print("state:", extended_state)

        selected_action = erm_opt_policy[t, extended_state["state"]]

        if DEBUG:
            print("action:", selected_action)

        # Environment step.
        extended_state, cost, terminated = env.step(extended_state, selected_action)
        cumulative_discounted_cost += cost * env.mdp["gamma"]**t

        if DEBUG:
            print("cost:", cost)

    print("final discounted cumulative cost:", cumulative_discounted_cost)

    return cumulative_discounted_cost


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate environment.
    env = get_env(cfg["env"], cfg["H"])

    mcts_f_val = simulate_ERM_backward_induction(env=env,
                               H=cfg["H"],
                               erm_beta=cfg["erm_beta"])

    return mcts_f_val


def main(cfg):

    # Setup experiment data folder.
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "erm-backward-induction",
                                'gamma': MDPs[cfg['env']]['gamma'],
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
    exp_data["env"] = cfg['env']

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print(exp_name)

    return exp_name


if __name__ == "__main__":
    main(cfg = CONFIG)
