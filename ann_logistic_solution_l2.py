
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

# --- Activation functions --- #
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    z = sigmoid(x)
    return z * (1 - z)

# --- Cost function with L2 Regularization --- #
def CostFcnLog_L2(theta, P0, n, t, per, tr, lambda_reg=0.01):
    H = int(len(theta) / 3)
    theta = theta.reshape(3, H)
    w, u, v = theta[0], theta[1], theta[2]

    net_input = tr @ w.reshape(1, -1) + u
    N = sigmoid(net_input) @ v.reshape(-1, 1)
    P = P0 + (tr - per * (t - 1)) * N
    dN_dt = dsigmoid(net_input) @ (v * w).reshape(-1, 1)
    dP_dt = N + (tr - per * (t - 1)) * dN_dt
    f = P * (1 - P)
    A = dP_dt - f

    reg_residuals = np.sqrt(lambda_reg) * theta.flatten()
    return np.concatenate((A.flatten(), reg_residuals))

# --- Main parameters --- #
iv = [0.1, 0.25, 0.75, 1.0, 1.5, 2.0]
T = 7
nr = 20
per = T / nr
pas = 100
n = 15
epochs = 4
H = 3
lambda_reg = 0.01

fig1, ax1 = plt.subplots()
LearnRates_all = []
CostLog = []

for i, P0_init in enumerate(iv):
    P0 = P0_init
    z0 = P0_init
    LearnRates = np.zeros((3 * (epochs + 1), H))
    cost_history = []

    for t in range(1, nr + 1):
        np.random.seed(t + i * nr)
        w = 0.1 * np.random.rand(H)
        u = 0.1 * np.random.rand(H)
        v = 0.1 * np.random.rand(H)
        theta = np.concatenate([w, u, v])
        LearnRates[0:3, :] = np.vstack([w, u, v])

        tr = np.linspace((t - 1) * per, t * per, n).reshape(-1, 1)
        tspan = np.linspace((t - 1) * per, t * per, pas + 1)

        for ep in range(epochs):
            result = least_squares(
                lambda th: CostFcnLog_L2(th, P0, n, t, per, tr, lambda_reg),
                theta,
                method='lm',
                max_nfev=10000,
                xtol=1e-4,
                ftol=1e-4,
                gtol=1e-4
            )
            theta = result.x
            LearnRates[3 * (ep + 1):3 * (ep + 2), :] = theta.reshape(3, H)

            residuals = CostFcnLog_L2(theta, P0, n, t, per, tr, lambda_reg)
            cost = np.sum(residuals**2)
            cost_history.append(cost)

        w, u, v = theta[:H], theta[H:2*H], theta[2*H:]
        net_input = np.outer(tspan, w) + u
        N = sigmoid(net_input) @ v
        P = P0 + (tspan - per * (t - 1)) * N

        def logistic(t, P): return P * (1 - P)
        sol = solve_ivp(logistic, [tspan[0], tspan[-1]], [z0], t_eval=tspan)
        z0 = sol.y[0, -1]
        P0 = P[-1]

        ax1.plot(tspan, P, 'b-', linewidth=1)
        ax1.plot(sol.t, sol.y[0], 'r--', linewidth=1)

        if t == 1:
            SelectedLearnRates = LearnRates.copy()
            SelectedCosts = cost_history.copy()

    LearnRates_all.append(LearnRates)
    CostLog.append(cost_history)

ax1.set_xlabel('t (time)', fontsize=14, fontweight='bold')
ax1.set_ylabel('P (population)', fontsize=14, fontweight='bold')
ax1.legend(['ANN solution', 'R-K solution'])
plt.title("ANN vs Runge-Kutta Solution", fontsize=14)
plt.grid(True)
plt.show()

# --- Plot Learning Rates (for t=1) --- #
LearnRates = SelectedLearnRates
epochs_range = list(range(epochs + 1))

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(epochs_range, LearnRates[0::3, :], '-*', linewidth=2)
axs[0].set_title('Learning rates (w_h)', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Values of coefficients', fontsize=10, fontweight='bold')
axs[0].legend([f'w_{i+1}' for i in range(H)], loc='lower left', fontsize=8)

axs[1].plot(epochs_range, LearnRates[1::3, :], '-*', linewidth=2)
axs[1].set_title('Learning rates (u_h)', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Values of coefficients', fontsize=10, fontweight='bold')
axs[1].legend([f'u_{i+1}' for i in range(H)], loc='lower left', fontsize=8)

axs[2].plot(epochs_range, LearnRates[2::3, :], '-*', linewidth=2)
axs[2].set_title('Learning rates (v_h)', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Iteration', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Values of coefficients', fontsize=10, fontweight='bold')
axs[2].legend([f'v_{i+1}' for i in range(H)], loc='lower left', fontsize=8)
plt.tight_layout()
plt.show()

# --- Plot Cost per Epoch for t=1 --- #
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), SelectedCosts, 'o-', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Cost', fontsize=12, fontweight='bold')
plt.title('Cost per Epoch (t = 1)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
