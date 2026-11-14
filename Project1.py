import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Define swish activation manually (if not using TensorFlow 2.4+)
def swish(x):
    return x * tf.math.sigmoid(x)


# Prepare input (centered)
x = np.linspace(0, 2, 100).reshape(-1, 1)
x_centered = x - 1.0
x_tf = tf.constant(x_centered, dtype=tf.float32)

# Define deeper neural network with swish activation
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  
    tf.keras.layers.Dense(40, activation=swish),
    tf.keras.layers.Dense(40, activation=swish),
    tf.keras.layers.Dense(40, activation=swish),
    tf.keras.layers.Dense(1)
])



# Trial solution: y_t(x) = 1 + x * NN(x), satisfying y(0) = 1
def trial_solution(x_val):
    return 1 + (x_val + 1.0) * model(x_val)


# Loss function: MSE of residual
def loss_fn():
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        y_t = trial_solution(x_tf)
        dy_dx = tape.gradient(y_t, x_tf)
        x_orig = x_tf + 1.0
        residual = dy_dx + y_t - tf.exp(-x_orig)
        return tf.reduce_mean(tf.square(residual))


# Optimizer with manual learning rate adjustment
initial_lr = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

# Training loop with LR decay
for i in range(5000):
    if i == 1000:
        optimizer.learning_rate.assign(0.001)
    if i == 3000:
        optimizer.learning_rate.assign(0.0003)

    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if i % 500 == 0:
        print(f"Step {i}, Loss: {loss.numpy()}")

# Evaluate solution
x_test = np.linspace(0, 2, 100).reshape(-1, 1)
x_test_centered = x_test - 1.0
x_test_tf = tf.constant(x_test_centered, dtype=tf.float32)

y_pred = trial_solution(x_test_tf).numpy()
y_exact = np.exp(-x_test)

# Plot approximation vs. exact
plt.plot(x_test, y_pred, label="ANN Approximation")
plt.plot(x_test, y_exact, '--', label="Exact Solution")
plt.title("Improved Gradient Differential Approximation via ANN")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Plot residual
with tf.GradientTape() as tape:
    tape.watch(x_test_tf)
    y_test_t = trial_solution(x_test_tf)
    dy_dx_test = tape.gradient(y_test_t, x_test_tf)
    residual_test = dy_dx_test + y_test_t - tf.exp(-x_test_tf - 1.0)

plt.plot(x_test, residual_test.numpy(), label="Residual")
plt.title("Residual of Differential Equation")
plt.xlabel("x")
plt.ylabel("Residual")
plt.grid(True)
plt.show()
