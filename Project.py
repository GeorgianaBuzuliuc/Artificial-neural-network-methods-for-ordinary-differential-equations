import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# 1. Încarcă fișierul Excel
df = pd.read_excel("HousePricePredictionResults.xlsx")

# 2. Folosim Predicted Price ca input și Actual Price ca target
X = df[['Predicted Price']].values  # input feature
y = df[['Actual Price']].values     # target value

# 3. Normalizare min-max
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. Împărțire train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 5. Definim rețeaua FFNN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 6. Antrenare
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.1, verbose=1)

# 7. Prezicere
y_pred_scaled = model.predict(X_test)
y_pred_dollars = scaler_y.inverse_transform(y_pred_scaled)
y_test_dollars = scaler_y.inverse_transform(y_test)

# 8. Afișare rezultate în terminal
results = pd.DataFrame({
    'Actual Price (USD)': y_test_dollars.flatten(),
    'Predicted Price (ANN, USD)': y_pred_dollars.flatten()
})
print(results.head(102))


# 9. Evaluare performanță
mse = mean_squared_error(y_test_dollars, y_pred_dollars)
r2 = r2_score(y_test_dollars, y_pred_dollars)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Ploturi
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test_dollars.flatten(), y=y_pred_dollars.flatten(), alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price (ANN)")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.show()
