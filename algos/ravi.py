import cvxpy as cp
import numpy as np


def solve_risk_averse(MDP, interpolation_points=10):
    """
    This function performes value iteration using the CVaR Bellman operator .

    Args:
        MDP: (X,A,p0,C,P,gamma): MDP to solve with risk aware value iteration.
        interpolation_points (int): Specifies the number of interpolation points along the augmented state dimension.

    Returns:
        V*: Value vector, V*[x,y] = value of state x for alpha y. !!! In this code y = 1 - our_alpha

    To get the optimal policy from V*, we evaluate Q*[x,y_t] = min_a { C[x,a] + gamma * max_xi { min(interp_funct[y_{t-1}*xi]) @ P[a][x,:] / y_{t-1} } }

    """
    X = MDP[0]
    A = MDP[1]
    C = MDP[3]
    P = MDP[4]
    g = MDP[5]
    N = np.linspace(0, 1, interpolation_points)
    V = np.zeros((len(X), len(N)))
    V_temp = np.zeros((len(X), len(N)))
    delta = 99

    while delta > 10e-2:
        delta = 0

        for x in X:
            for j, y in enumerate(N):
                xi = cp.Variable(len(X))
                interpolation_functions = [
                    N[i] * V[:, i]
                    + cp.multiply(
                        (N[i + 1] * V[:, i + 1] - N[i] * V[:, i]) / (N[i + 1] - N[i]),
                        (xi - N[i]),
                    )
                    for i in range(len(N) - 1)
                ]

                temp_q = np.zeros(len(A))

                for a in A:
                    if y == 0:
                        temp_q[a] = max(V[:, 0] * (P[a][x, :] > 0))
                    else:
                        objective = cp.Maximize(
                            P[a][x, :] @ cp.minimum(*interpolation_functions) / y
                        )
                        constraint = [
                            xi >= np.zeros(len(X)),
                            xi <= np.ones(len(X)),
                            P[a][x, :] @ xi == y,
                        ]
                        prob = cp.Problem(objective, constraint)
                        prob.solve(verbose=False)
                        temp_q[a] = prob.value

                V_temp[x, j] = min(C[x, :] + g * temp_q)

        delta = max(delta, abs(V_temp - V).max())
        V = V_temp.copy()
        print("delta:", delta)

    return V


def get_action(MDP, V, x, y, interpolation_points):
    X = MDP[0]
    A = MDP[1]
    C = MDP[3]
    P = MDP[4]
    g = MDP[5]
    N = np.linspace(0, 1, interpolation_points)
    xi = cp.Variable(len(X))
    interpolation_functions = [
        N[i] * V[:, i]
        + cp.multiply(
            (N[i + 1] * V[:, i + 1] - N[i] * V[:, i]) / (N[i + 1] - N[i]), (xi - N[i])
        )
        for i in range(len(N) - 1)
    ]

    temp_q = np.zeros(len(A))
    temp_y = np.zeros((len(A), len(X)))

    for a in A:
        if np.isclose(y, 0, atol=1e-05):
            temp_q[a] = max(V[:, 0] * (P[a][x, :] > 0))
        else:
            objective = cp.Maximize(
                P[a][x, :] @ cp.minimum(*interpolation_functions) / y
            )
            constraint = [
                xi >= np.zeros(len(X)),
                xi <= np.ones(len(X)),
                P[a][x, :] @ xi == y,
            ]
            prob = cp.Problem(objective, constraint)
            prob.solve(verbose=False)
            temp_q[a] = prob.value
            temp_y[a, :] = xi.value

    action = np.argmin(C[x, :] + g * temp_q)
    return action, temp_y[action]
