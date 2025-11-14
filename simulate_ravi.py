import json
import multiprocessing as mp
import os
import pathlib
from datetime import datetime

import numpy as np
from tqdm import tqdm

from envs.envs import get_env
from algos.ravi import solve_risk_averse, get_action

DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent) + '/data/'
print(DATA_FOLDER_PATH)

CONFIG = {
    "N": 100, # Number of rollouts to run.
    "env": "linear_mdp",
    "H": 100, # Truncation length.
    "interp_points": 50,
}

def create_exp_name(args: dict) -> str:
    return args['env'] + '_' + args['algo'] + '_gamma_' + str(args['gamma']) + '_' + \
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

def main(cfg):

    # Setup experiment data folder.
    env = get_env(cfg["env"])
    exp_name = create_exp_name({'env': cfg['env'],
                                'algo': "ravi",
                                'gamma': env['gamma'],
                                })
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)
    print('Config:')
    print(cfg)

    # Simulate.
    print('\nSimulating...')

    env = get_env(cfg["env"])
    print("env", env)

    # Instantiate MDP.
    mdp = (
        env["states"],
        env["actions"],
        env["p_0"],
        env["C"],
        env["P"],
        env["gamma"],
    )

    value = solve_risk_averse(mdp, interpolation_points=cfg["interp_points"])
    norm = (1 - env["gamma"]) / (1-env["gamma"]**cfg["H"])

    f_vals_alphas = {}

    for cvar_alpha in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:

        print("cvar_alpha", cvar_alpha)

        f_vals = []
        for _ in range(cfg["N"]):

            occupancy = np.zeros((len(env["states"]), len(env["actions"])))
            state = np.random.choice(mdp[0], p=env["p_0"])
            cvar = cvar_alpha

            j = 0
            while j < cfg["H"]:

                # if state < mdp[0].shape[0] - 1:
                action, temp_y = get_action(mdp, value, int(state), cvar, cfg["interp_points"])
                    
                occupancy[state, action] += env["gamma"] ** j
                state = np.random.choice(mdp[0], p=mdp[4][action][state])
                cvar = temp_y[state]

                if np.isclose(cvar, 0.0):
                    cvar = 0.0
                
                j += 1

            occupancy *= norm
            assert np.isclose(np.sum(occupancy), 1.0), np.sum(occupancy)
            f_vals.append(np.sum(env["C"] * occupancy))

        f_vals_alphas[cvar_alpha] = np.array(f_vals)

    exp_data = {}
    exp_data["config"] = cfg
    exp_data["f_vals_alphas"] = f_vals_alphas
    exp_data["env"] = env
    exp_data["env"]["f"] = None

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    return exp_name


if __name__ == "__main__":
    main(cfg = CONFIG)
