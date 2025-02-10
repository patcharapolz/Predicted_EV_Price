import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

file_path = "Project/data/EV_cars_Edited.csv"
columns = ["Battery", "Top_speed", "Acceleration", "Efficiency", "Fast_charge", "Range", "Price"]
df = pd.read_csv(file_path)
df = df[columns]
df = df.fillna(df.mean())

X = df.drop(columns=["Price"]).values
y = df["Price"].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.yticks([1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000], 
           ["1,000,000", "2,000,000", "3,000,000", "4,000,000", 
            "5,000,000", "6,000,000", "7,000,000", "8,000,000"])
plt.plot(y_test_real, label="Actual Price", color="red")
plt.plot(y_pred, label="Predicted Price", color="blue")
plt.xlabel("Sample")
plt.ylabel("Price (€)")
plt.title("Actual vs Predicted Price")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()


mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} €")
print(f"R² Score: {r2:.4f}")