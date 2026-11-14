import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fix reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Parametri
a_val = 1.0
t0 = 0.0
x0 = 0.0

# Domeniu de lucru
t_test = np.linspace(0, 6 * np.pi, 300).reshape(-1, 1)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)

t_train = np.linspace(0, 6 * np.pi, 200).reshape(-1, 1)
t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)

# Definirea modelului ANN
class FFNN(tf.keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.out = tf.keras.layers.Dense(1)

    def call(self, t):
        z = self.hidden1(t)
        z = self.hidden2(z)
        return self.out(z)

# Soluție trial
def trial_solution(t, model):
    return x0 + (t - t0) * model(t)

# Rezidualul ODE-ului
def compute_residual(model, t, a_val):
    with tf.GradientTape() as tape:
        tape.watch(t)
        x_hat = trial_solution(t, model)
    dx_dt = tape.gradient(x_hat, t)
    residual = dx_dt + a_val * tf.math.tanh(t) * x_hat - tf.math.cos(t)
    return residual

# Antrenarea rețelei
model = FFNN(hidden_units=50)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for epoch in range(12000):
    with tf.GradientTape() as tape:
        residuals = compute_residual(model, t_train_tf, a_val)
        loss = tf.reduce_mean(tf.square(residuals))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 600 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Predicție ANN
x_ann = trial_solution(t_test_tf, model).numpy()

# Soluția exactă cu solve_ivp
def ode_system(t, x):
    return -a_val * np.tanh(t) * x + np.cos(t) *

sol = solve_ivp(ode_system, [t0, float(t_test[-1])], [x0], t_eval=t_test.flatten())
x_exact = sol.y[0]

# Plot: soluție ANN vs soluție exactă
plt.figure(figsize=(10, 6))
plt.plot(t_test, x_ann, label="ANN Approximation", color="blue")
plt.plot(t_test, x_exact, label="Exact Solution", linestyle="dashed", color="red")
plt.title("ANN Approximation vs Exact Solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.show()

# Compute errors
mse = mean_squared_error(x_exact, x_ann)
mae = mean_absolute_error(x_exact, x_ann)
max_error = np.max(np.abs(x_exact - x_ann))

# Print error metrics
print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Maximum Absolute Error: {max_error:.6f}")
