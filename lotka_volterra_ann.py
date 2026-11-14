import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def cost_function(theta, tr, x0, y0, alpha, t, per, n, H):
    theta = theta.reshape(6, H)
    w1, u1, v1 = theta[0], theta[1], theta[2]
    w2, u2, v2 = theta[3], theta[4], theta[5]

    tr = tr.reshape(-1, 1)
    N1 = sigmoid(tr @ w1.reshape(1, -1) + u1) @ v1.reshape(-1, 1)
    N2 = sigmoid(tr @ w2.reshape(1, -1) + u2) @ v2.reshape(-1, 1)

    x = x0 + (tr - per * (t - 1)) * N1
    y = y0 + (tr - per * (t - 1)) * N2

    dN1_dt = dsigmoid(tr @ w1.reshape(1, -1) + u1) @ ((v1 * w1).reshape(-1, 1))
    dN2_dt = dsigmoid(tr @ w2.reshape(1, -1) + u2) @ ((v2 * w2).reshape(-1, 1))

    dx_dt = N1 + (tr - per * (t - 1)) * dN1_dt
    dy_dt = N2 + (tr - per * (t - 1)) * dN2_dt

    fx = x - x * y
    fy = alpha * (x * y - y)

    A = dx_dt - fx
    B = dy_dt - fy

    cost = np.sum(A ** 2 + B ** 2) / n
    return cost

start_time = time.time()
x0, y0 = 1.25, 0.66
alpha = 1.0
T = 10
nr = 100
pas = 1e2
tspan_full = np.linspace(0, T, int(T * pas) + 1)
z10, z20 = x0, y0
per = T / nr
n = 10
epochs = 10
H = 3
LearnRates = np.zeros((6 * epochs, H))

all_x, all_y, all_z1, all_z2, all_t = [], [], [], [], []

for t in range(1, nr + 1):
    w1 = np.random.rand(H)
    u1 = np.random.rand(H)
    v1 = np.random.rand(H)
    w2 = np.random.rand(H)
    u2 = np.random.rand(H)
    v2 = np.random.rand(H)
    theta = np.array([w1, u1, v1, w2, u2, v2])
    LearnRates[0:6, :] = theta

    tr = np.linspace((t - 1) * per, t * per, n)
    tspan = np.linspace((t - 1) * per, t * per, int(per * pas) + 1)

    for ep in range(epochs):
        res = least_squares(cost_function, theta.flatten(),
                            args=(tr, x0, y0, alpha, t, per, n, H),
                            method='trf')
        theta = res.x.reshape(6, H)
        LearnRates[6 * ep:6 * (ep + 1), :] = theta

    w1, u1, v1 = theta[0], theta[1], theta[2]
    w2, u2, v2 = theta[3], theta[4], theta[5]

    tspan_input = tspan.reshape(-1, 1)
    N1 = sigmoid(tspan_input @ w1.reshape(1, -1) + u1) @ v1.reshape(-1, 1)
    N2 = sigmoid(tspan_input @ w2.reshape(1, -1) + u2) @ v2.reshape(-1, 1)

    x = x0 + (tspan_input - per * (t - 1)) * N1
    y = y0 + (tspan_input - per * (t - 1)) * N2

    all_x.append(x.flatten())
    all_y.append(y.flatten())
    all_t.append(tspan)

    def lotka_volterra(t, z):
        return [z[0] - z[0] * z[1], alpha * (z[0] * z[1] - z[1])]

    sol = solve_ivp(lotka_volterra, [tspan[0], tspan[-1]], [z10, z20], t_eval=tspan)
    z10, z20 = sol.y[0, -1], sol.y[1, -1]
    x0, y0 = x[-1, 0], y[-1, 0]

    all_z1.append(sol.y[0])
    all_z2.append(sol.y[1])

#Plot
plt.figure()
# Plot ANN solution (only once with label)
plt.plot(all_x[0], all_y[0], 'b-', linewidth=1, label='ANN solution')
for x, y in zip(all_x[1:], all_y[1:]):
    plt.plot(x, y, 'b-', linewidth=1)

# Plot R-K solution (only once with label)
plt.plot(all_z1[0], all_z2[0], 'r--', linewidth=1, label='R-K solution')
for z1, z2 in zip(all_z1[1:], all_z2[1:]):
    plt.plot(z1, z2, 'r--', linewidth=1)
plt.xlabel('x (prey)')
plt.ylabel('y (predator)')
plt.title('Phase Plane')
plt.legend(['ANN solution', 'R-K solution'])
plt.grid(True)

plt.figure()
plt.subplot(1, 2, 1)
for x, z1, t in zip(all_x, all_z1, all_t):
    plt.plot(t, x, 'b-')
    plt.plot(t, z1, 'r--')
plt.xlabel('t (time)')
plt.ylabel('x (prey)')

plt.subplot(1, 2, 2)
for y, z2, t in zip(all_y, all_z2, all_t):
    plt.plot(t, y, 'b-')
    plt.plot(t, z2, 'r--')
plt.xlabel('t (time)')
plt.ylabel('y (predator)')

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
for j in range(H):
    plt.plot(np.arange(epochs), LearnRates[0::6, j], '-*')
plt.title('Learning rates (w_h)')
plt.ylabel('Values')

plt.subplot(3, 1, 2)
for j in range(H):
    plt.plot(np.arange(epochs), LearnRates[1::6, j], '-*')
plt.title('Learning rates (u_h)')
plt.ylabel('Values')

plt.subplot(3, 1, 3)
for j in range(H):
    plt.plot(np.arange(epochs), LearnRates[2::6, j], '-*')
plt.title('Learning rates (v_h)')
plt.xlabel('Epoch')
plt.ylabel('Values')

plt.tight_layout()
plt.show()
print(f"Execution completed in {time.time() - start_time:.2f} seconds.")
