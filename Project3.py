import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Exact solution
def y_exact(t):
    return (1/4) * t**4 - tf.math.log(t) - 0.25

# Neural network with two hidden layers and regularization
class ODEModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_units, activation='swish',
                                             kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.hidden2 = tf.keras.layers.Dense(hidden_units, activation='swish',
                                             kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.output_layer = tf.keras.layers.Dense(1)

        # Parametru trainabil alpha pentru log(t)
        self.alpha = tf.Variable(1.0, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.output_layer(x)

# Trial solution: includes log(t) term and enforces y(1) = 0
def trial_solution(t_raw, model_output, model):
    log_term = tf.math.log(t_raw)
    return (t_raw - 1.0) * (model_output + model.alpha * log_term)

# Residual calculation
def compute_residual(model, t_raw, t_input):
    with tf.GradientTape() as tape:
        tape.watch(t_raw)
        model_output = model(t_input)
        y_pred = trial_solution(t_raw, model_output, model)
    dy_dt = tape.gradient(y_pred, t_raw)
    return dy_dt - (t_raw**3 - 1.0 / t_raw)

# Training function
def train(model, t_raw, t_input, epochs=5000):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=4000,
        decay_rate=0.85
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    weight_log = {"w": [], "u": [], "v": []}

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            residuals = compute_residual(model, t_raw, t_input)
            loss = tf.reduce_mean(tf.square(residuals))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        w = model.output_layer.kernel.numpy().flatten()
        u = model.hidden1.kernel.numpy().flatten()
        b = model.hidden1.bias.numpy().flatten()
        weight_log["w"].append(w.copy())
        weight_log["u"].append(u.copy())
        weight_log["v"].append(b.copy())

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

    return weight_log

# Plot solution
def plot_solution(model):
    t_test = tf.reshape(tf.linspace(0.25, 2.0, 200), (-1, 1))
    t_scaled = 2 * (t_test - tf.reduce_mean(t_test)) / (tf.reduce_max(t_test) - tf.reduce_min(t_test))
    log_term = tf.math.log(t_test)
    t_input = tf.concat([t_scaled, log_term], axis=1)

    model_output = model(t_input)
    y_pred = trial_solution(t_test, model_output, model)
    y_true = y_exact(t_test)
    print(f"Learned alpha (log term scale): {model.alpha.numpy():.6f}")
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    print(f"Final MSE on test data: {mse.numpy():.6f}")
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    print(f"Final MAE on test data: {mae.numpy():.6f}")
    max_error = tf.reduce_max(tf.abs(y_true - y_pred))
    print(f"Final Max Error on test data: {max_error.numpy():.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(t_test.numpy(), y_true.numpy(), label="Exact solution", linestyle='dashed')
    plt.plot(t_test.numpy(), y_pred.numpy(), label="ANN Approximation")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Solving $dy/dt = t^3 - 1/t$, $y(1)=0$ with ANN (swish, log(t) input)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot residual
def plot_residuals(model, t_raw, t_input):
    residuals = compute_residual(model, t_raw, t_input)
    plt.figure(figsize=(8, 4))
    plt.plot(t_raw.numpy(), residuals.numpy())
    plt.title("Residual of the ODE over the domain")
    plt.xlabel("t")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot weights
def plot_weights(weight_log):
    epochs = len(weight_log["w"])
    H = len(weight_log["w"][0])
    x = np.arange(epochs)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for h in range(H):
        axs[0].plot(x, [w[h] for w in weight_log["w"]], label=f"w{h+1}")
    axs[0].set_title("Weights w (output layer)")

    for h in range(H):
        axs[1].plot(x, [u[h] for u in weight_log["u"]], label=f"u{h+1}")
    axs[1].set_title("Weights u (input to hidden)")

    for h in range(H):
        axs[2].plot(x, [v[h] for v in weight_log["v"]], label=f"v{h+1}")
    axs[2].set_title("Biases v (hidden layer bias)")

    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Domeniul inițial [0.25, 2.0]
    t_base = tf.linspace(0.25, 2.0, 300)

    # Mai mulți puncte în [1.5, 2.0]
    t_focus = tf.linspace(1.5, 2.0, 200)

    # Combină și sortează toate punctele
    t_combined = tf.concat([t_base, t_focus], axis=0)
    t_combined = tf.sort(t_combined)

    t_raw = tf.reshape(t_combined, (-1, 1))
    t_scaled = 2 * (t_raw - tf.reduce_mean(t_raw)) / (tf.reduce_max(t_raw) - tf.reduce_min(t_raw))
    log_term = tf.math.log(t_raw)
    t_input = tf.concat([t_scaled, log_term], axis=1)

    model = ODEModel(hidden_units=20)
    weight_log = train(model, t_raw, t_input, epochs=5000)
    plot_solution(model)
    plot_residuals(model, t_raw, t_input)
    plot_weights(weight_log)

if __name__ == "__main__":
    main()
