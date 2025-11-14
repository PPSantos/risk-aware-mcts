import json
import multiprocessing as mp
import os
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from algos.erm_mcts import ERMMCTS
from envs.envs import Occupancy_MDP, get_env

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

CONFIG = {
    "N": 1, # Number of experiments to run.
    "num_processors": 1,
    "env": "four_state_mdp",
    "H": 100, # Truncation length.
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

def simulate_ERM_MCTS(mdp, H, erm_beta, n_iter_per_timestep=1_000):

    # Instantiate environment.
    env = get_env(mdp, H)

    # Sample initial state from the extended MDP.
    extended_state = env.sample_initial_state()

    # Simulate until termination.
    cumulative_discounted_cost = 0.0
    for t in tqdm(range(H)):

        mcts = ERMMCTS(initial_state=extended_state, env=env, K_ucb=np.sqrt(2),
                        erm_beta=erm_beta, rollout_policy=None)
        mcts.learn(n_iters=n_iter_per_timestep)
        selected_action = mcts.best_action()

        # Environment step.
        extended_state, cost, terminated = env.step(extended_state, selected_action)
        cumulative_discounted_cost += cost * mdp["gamma"]*t

    print("final discounted cumulative cost:", cumulative_discounted_cost)

    return cumulative_discounted_cost


def run(cfg, seed):

    print('Running seed=', seed)

    np.random.seed(seed)

    # Instantiate MDP.
    env = get_env(cfg["env"])
    print("env", env)

    mcts_f_val = simulate_ERM_MCTS(mdp=env,
                               H=cfg["H"],
                               erm_beta=cfg["erm_beta"],
                               n_iter_per_timestep=cfg["n_iter_per_timestep"])

    return mcts_f_val


def main(cfg):

    # Setup experiment data folder.
    env = get_env(cfg["env"])
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "mcts",
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
