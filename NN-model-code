import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1️⃣ Load dataset
df = pd.read_csv("bollywood_box_office_dataset.csv")

# 2️⃣ Select features and target
features = ["Budget", "Screen Count", "Director Success", "Lead Actor Popularity", "Past Success Score", "Music Popularity"]
target = "BoxOfficeCollection"

X = df[features].values
y = df[target].values.reshape(-1, 1)  # Ensure target is column vector

# 3️⃣ Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4️⃣ Build the Neural Network Model
model = Sequential([
    Dense(10, input_dim=6, activation='relu'),  # Hidden Layer 1
    Dense(8, activation='relu'),  # Hidden Layer 2
    Dense(1, activation='linear')  # Output Layer
])

# 5️⃣ Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6️⃣ Train the Model
model.fit(X_scaled, y_scaled, epochs=500, batch_size=16, verbose=1)

# 7️⃣ Make Predictions on a New Movie
test_movie = np.array([[100, 2000, 80, 90, 75, 85]])  # Example input values
test_movie_scaled = scaler_X.transform(test_movie)

predicted_bo_scaled = model.predict(test_movie_scaled)
predicted_bo = scaler_y.inverse_transform(predicted_bo_scaled)

print("Predicted Box Office Collection:", predicted_bo[0][0], "Crores")
