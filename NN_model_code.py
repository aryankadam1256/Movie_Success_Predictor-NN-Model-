import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ✅ 1️⃣ Set Seed for Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ✅ 2️⃣ Load dataset
df = pd.read_csv("E:\MACHINE LEARNING\Advanced_Learning_NNs\Movie_Success_Predictor\DATASET\movie_dataset_final.csv")

# ✅ 3️⃣ Clean Column Names
df.columns = df.columns.str.strip()

# ✅ 4️⃣ Select Features and Target
features = ["Budget (Cr)", "Screens", "Director Success", "Lead Actor Popularity", "Past Success Score", "Music Popularity"]
target = "Box Office (Cr)"

X = df[features].values
y = df[target].values.reshape(-1, 1)  # Keep target as column vector

# ✅ 5️⃣ Normalize Features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# ✅ 6️⃣ Build Neural Network Model
model = Sequential([
    Dense(16, input_dim=6, activation='relu', kernel_initializer='he_normal'),  
    Dropout(0.2),
    Dense(12, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.1),
    Dense(8, activation='relu'),
    Dense(1, activation='relu')
])

# ✅ 7️⃣ Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
              loss='mean_absolute_error',
              metrics=['mae'])

# ✅ 8️⃣ Callbacks for Stability
early_stopping = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=15, min_lr=1e-5)

# ✅ 9️⃣ Train Model
history = model.fit(X_scaled, y, epochs=500, batch_size=16, verbose=1, callbacks=[early_stopping, reduce_lr])

# # ✅ 🔟 Print Model Weights & Biases
# print("\n🔹 MODEL WEIGHTS & BIASES 🔹")
# for i, layer in enumerate(model.layers):
#     if isinstance(layer, Dense):  # Only extract weights from Dense layers
#         weights, biases = layer.get_weights()
#         print(f"\n🟢 Layer {i+1} Weights:\n", np.array(weights).tolist())  # Convert to list for readability
#         print(f"\n🔴 Layer {i+1} Biases:\n", np.array(biases).tolist())

# ✅ 🔟 Predict for New Movie
test_movie = np.array([[300, 8000, 90, 90, 100, 90]])  # Example input values
test_movie_scaled = scaler_X.transform(test_movie)

predicted_bo = model.predict(test_movie_scaled)

print("\n🎬 Predicted Box Office Collection:", round(predicted_bo[0][0], 2), "Crores")
