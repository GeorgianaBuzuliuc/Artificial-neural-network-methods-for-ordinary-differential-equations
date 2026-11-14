#   Use ANN to approximate solutions to IVP: x' = - tanh(t) * x + cos(t), x(0) = 0
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#   Fix reproducibility for consistent results
np.random.seed(1)
tf.random.set_seed(1)

#   Initial condition
x0 = 0.0
t0 = 0.0

#   NN model
class FFNN(tf.keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        # Three hidden layers with tanh activation
        self.hidden1 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        # Output layer with a single neuron (for x(t) prediction)
        self.out = tf.keras.layers.Dense(1)

    def call(self, t):
        if t.shape.rank == 1:
            t = tf.expand_dims(t, axis=-1)
        z = self.hidden1(t)
        z = self.hidden2(z)
        z = self.hidden3(z)
        return self.out(z)

#   Trial solution (it incorporates the initial condition for each subinterval)
def trial_solution_interval(t_input, model_instance, current_x_initial, current_t_initial):
    return current_x_initial + (t_input - current_t_initial) * model_instance(t_input)

#   ODE residual
def compute_residual_interval(model_instance, t_input, current_x_initial, current_t_initial):
    with tf.GradientTape() as tape:
        tape.watch(t_input)  # Watch 't_input' to compute gradients w.r.t. it (dx/dt)
        x_hat = trial_solution_interval(t_input, model_instance, current_x_initial, current_t_initial)

    dx_dt = tape.gradient(x_hat, t_input)  # Compute dx_hat/dt

    # ODE: x' = -tanh(t)*x + cos(t)  => Residual = x' + tanh(t)*x - cos(t)
    residual = dx_dt + tf.math.tanh(t_input) * x_hat - tf.math.cos(t_input)
    return residual

#   The exact solution
def exact_solution_func(t):
    return 0.5 * (np.sin(t) + np.cos(t) * np.tanh(t))

#    Global time domain setup
t_total_end = 4 * np.pi
num_test_points_global = 100  # Number of points for the overall comparison plot
t_test_global = np.linspace(t0, t_total_end, num_test_points_global).reshape(-1, 1)

#   Subinterval configuration
num_intervals = 3  # The total time range [0, 5*pi] will be split into 10 subintervals
interval_length = (t_total_end - t0) / num_intervals

#   Lists to store the approximate and exact solutions for each subinterval
x_ann_intervals = []
t_ann_intervals = []
x_exact_intervals = []
t_exact_intervals = []

#   Initial conditions for the first subinterval
current_x0_ann = x0
current_t0_ann = t0

#   Sequential training over subintervals
for i in range(num_intervals):
    print(f"\n--- Simulating interval {i + 1}/{num_intervals} ---")

    # Define the start and end times for the current subinterval
    t_start_interval = current_t0_ann
    t_end_interval = current_t0_ann + interval_length

    # Generate training points for the current subinterval
    num_points_per_interval_train = 200  # Number of training points within each subinterval
    t_train_interval = np.linspace(t_start_interval, t_end_interval, num_points_per_interval_train).reshape(-1, 1)
    t_train_interval_tf = tf.convert_to_tensor(t_train_interval, dtype=tf.float32)

    # Extract test points from the global test set that fall within the current subinterval
    t_test_interval_indices = np.where(
        (t_test_global >= t_start_interval - 1e-9) & (t_test_global <= t_end_interval + 1e-9))
    # Reshape to (N, 1) to ensure compatibility with Dense layers
    t_test_interval = t_test_global[t_test_interval_indices].reshape(-1, 1)
    t_test_interval_tf = tf.convert_to_tensor(t_test_interval, dtype=tf.float32)

    # Initialize a new FFNN model for each subinterval
    model = FFNN(hidden_units = 10)  # Number of neurons in hidden layers
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.05)

    tolerance = 0.0001  # Early stopping threshold: Stop if loss falls below this value
    max_epochs = 1000   # Maximum training epochs per subinterval

    # Training loop for the current subinterval
    for epoch in range(max_epochs):
        with tf.GradientTape() as tape:
            # Compute residual using the ANN's current initial condition for this segment
            residuals = compute_residual_interval(model, t_train_interval_tf, current_x0_ann, current_t0_ann)
            loss = tf.reduce_mean(tf.square(residuals))  # Loss is the MSE of the residual

        grads = tape.gradient(loss, model.trainable_variables)  # Compute gradients of loss w.r.t. model weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # Update model weights

        # Print progress and check for early stopping
        if epoch % 500 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch}, Loss: {loss.numpy():.6f}")

        if loss.numpy() < tolerance:
            print(f"  Training stopped at epoch {epoch} as loss ({loss.numpy():.6f}) is less than {tolerance}")
            break
    if loss.numpy() >= tolerance:
        print(f"  Training finished max epochs {max_epochs}, final loss: {loss.numpy():.6f}")

    # The ANN approximation for the current subinterval
    x_ann_interval = trial_solution_interval(t_test_interval_tf, model, current_x0_ann, current_t0_ann).numpy()
    x_ann_intervals.append(x_ann_interval)
    t_ann_intervals.append(t_test_interval)

    # The exact solution for the current subinterval
    x_exact_interval = exact_solution_func(t_test_interval)
    x_exact_intervals.append(x_exact_interval)
    t_exact_intervals.append(t_test_interval)

    # Update initial conditions for the next subinterval
    if x_ann_interval.size > 0:  # Ensure there are points in the interval
        current_x0_ann = x_ann_interval[-1][0]  # Take the scalar value from the (1,1) array
    else:
        current_x0_ann = x_exact_intervals[-1][-1]
    current_t0_ann = t_end_interval  # The new start time is the end of the current interval

#   Concatenate all subinterval solutions
t_ann_joined = np.concatenate(t_ann_intervals).reshape(-1, 1)
x_ann_joined = np.concatenate(x_ann_intervals).reshape(-1, 1)
t_exact_joined = np.concatenate(t_exact_intervals).reshape(-1, 1)
x_exact_joined = np.concatenate(x_exact_intervals).reshape(-1, 1)

#   Plot the solutions
plt.figure(figsize=(12, 7))
plt.plot(t_ann_joined, x_ann_joined, label="ANN approximation", color="blue")
plt.plot(t_exact_joined, x_exact_joined, label="exact solution", linestyle="dashed", color="red")
plt.title("ANN approximation vs exact solution")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.show()

#   Compute and print evaluation metrics
mse_joined = mean_squared_error(x_exact_joined, x_ann_joined)
mae_joined = mean_absolute_error(x_exact_joined, x_ann_joined)
max_error_joined = np.max(np.abs(x_exact_joined - x_ann_joined))

print(f"\n--- Evaluation metrics ---")
print(f"Mean Squared Error (MSE): {mse_joined:.6f}")
print(f"Mean Absolute Error (MAE): {mae_joined:.6f}")
print(f"Maximum Absolute Error: {max_error_joined:.6f}")