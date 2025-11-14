import numpy as np

class ValueIteration(object):

    def __init__(self, mdp, gamma, epsilon=1e-04):
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

    def compute(self):

        print('Running value iteration...')
        Q_vals = np.zeros((len(self.mdp["states"]), len(self.mdp["actions"])))
        while True:
            Q_vals_old = np.copy(Q_vals)

            for state in self.mdp["states"]:
                for action in self.mdp["actions"]:

                    Q_next_states = np.dot(self.mdp["P"][action][state], np.min(Q_vals, axis=1))

                    Q_vals[state][action] = self.mdp["C"][state][action] + \
                        self.gamma * Q_next_states

            delta = np.sum(np.abs(Q_vals - Q_vals_old))
            print('Delta:', delta)

            if delta < self.epsilon:
                break

        return Q_vals
