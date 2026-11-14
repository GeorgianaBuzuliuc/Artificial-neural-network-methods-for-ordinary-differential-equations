import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Exact solution (for comparison)
def y_exact(t):
    return (1 / 4) * t ** 4 - tf.math.log(t) - 0.25


# PINN model (simple feedforward NN)
class PINN(tf.keras.Model):
    def __init__(self, hidden_units=20):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, t):
        x = self.hidden1(t)
        x = self.hidden2(x)
        return self.output_layer(x)


# Trial solution: automatically satisfies y(1) = 0
def trial_solution(t, model_output):
    return (t - 1.0) * model_output


# Residual of the ODE
def residual(t, model):
    with tf.GradientTape() as tape:
        tape.watch(t)
        y_pred = trial_solution(t, model(t))
    dy_dt = tape.gradient(y_pred, t)
    return dy_dt - (t ** 3 - 1 / t)


# Loss function: residual loss + boundary loss
def compute_loss(model, t_colloc, t_boundary):
    # Residual loss
    res = residual(t_colloc, model)
    res_loss = tf.reduce_mean(tf.square(res))

    # Boundary condition loss at t = 1 (y(1) = 0)
    y_b = trial_solution(t_boundary, model(t_boundary))
    bc_loss = tf.square(y_b)

    return res_loss + bc_loss


# Training
def train(model, epochs=5000):
    # Collocation and boundary points
    t_colloc = tf.reshape(tf.linspace(0.25, 2.0, 400), (-1, 1))
    t_boundary = tf.convert_to_tensor([[1.0]], dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    weight_log = {"w": [], "u": [], "v": []}

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, t_colloc, t_boundary)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        w = model.output_layer.kernel.numpy().flatten()
        u = model.hidden1.kernel.numpy().flatten()
        b = model.hidden1.bias.numpy().flatten()
        weight_log["w"].append(w.copy())
        weight_log["u"].append(u.copy())
        weight_log["v"].append(b.copy())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy().item():.6f}")

    return weight_log



# Plotting
def plot_solution(model):
    t_test = tf.reshape(tf.linspace(0.25, 2.0, 300), (-1, 1))
    y_pred = trial_solution(t_test, model(t_test))
    y_true = y_exact(t_test)

    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    max_error = tf.reduce_max(tf.abs(y_true - y_pred))

    print(f"\nFinal MSE: {mse.numpy():.6f}")
    print(f"Final MAE: {mae.numpy():.6f}")
    print(f"Final Max Error: {max_error.numpy():.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(t_test, y_true, label="Exact", linestyle="--")
    plt.plot(t_test, y_pred, label="PINN Approximation")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("PINN for $dy/dt = t^3 - 1/t$, $y(1)=0$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residual(model):
    t_res = tf.reshape(tf.linspace(0.25, 2.0, 300), (-1, 1))
    res = residual(t_res, model)
    plt.figure(figsize=(8, 4))
    plt.plot(t_res.numpy(), res.numpy())
    plt.title("Residual of the ODE over the domain")
    plt.xlabel("t")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weights(weight_log):
    epochs = len(weight_log["w"])
    H = len(weight_log["w"][0])
    x = np.arange(epochs)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for h in range(H):
        axs[0].plot(x, [w[h] for w in weight_log["w"]])
    axs[0].set_title("Weights w (output layer)")

    for h in range(H):
        axs[1].plot(x, [u[h] for u in weight_log["u"]])
    axs[1].set_title("Weights u (input to hidden)")

    for h in range(H):
        axs[2].plot(x, [v[h] for v in weight_log["v"]])
    axs[2].set_title("Biases v (hidden layer bias)")

    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

# Main run
def main():
    model = PINN(hidden_units=20)
    weight_log = train(model, epochs=5000)
    plot_solution(model)
    plot_residual(model)
    plot_weights(weight_log)


if __name__ == "__main__":
    main()
