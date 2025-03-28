# Movie_Success_Predictor-NN-Model-
For Bollywood Box Office Prediction, our model will follow these steps:

Load and preprocess the dataset

Read our CSV file containing Bollywood movie data

Normalize feature values (budget, screen count, etc.)

Define the Neural Network

Input Layer: Takes 6 features (Budget, Screen Count, etc.)

Hidden Layers: Capture non-linear patterns (like Budget x Actor Popularity effects)

Output Layer: Predicts Box Office Collection

Train the Model

Forward propagation

Loss calculation

Backpropagation to update weights

Evaluate & Predict

Test on some unseen data

Get box office predictions


 CODE EXPLANTION
1️⃣ Load and Preprocess Data
(Similar to Coffee Model - Load CSV, Normalize Data)

python
Copy
Edit
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("bollywood_box_office_dataset.csv")

# Select features and target
features = ["Budget", "Screen Count", "Director Success", "Lead Actor Popularity", "Past Success Score", "Music Popularity"]
target = "BoxOfficeCollection"

X = df[features].values
y = df[target].values.reshape(-1, 1)  # Ensure target is column vector

# Normalize the data (Scaling 0-1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
🔹 Why Normalize?

Just like in the Coffee Roaster model, NN works better when data is in a small range (0-1 or -1 to 1).

This prevents features with large numbers (like Budget in crores) from dominating smaller ones (like Music Popularity).

2️⃣ Define Neural Network Model
(Similar to Coffee Model - Layers, Weights, Activation)

python
Copy
Edit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the NN Model
model = Sequential([
    Dense(10, input_dim=6, activation='relu'),  # Hidden Layer 1 (10 neurons)
    Dense(8, activation='relu'),  # Hidden Layer 2 (8 neurons)
    Dense(1, activation='linear')  # Output Layer (Predict Box Office)
])

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
🔹 Explanation

Dense(10, input_dim=6, activation='relu') → First hidden layer with 10 neurons, taking 6 input features

Dense(8, activation='relu') → Second hidden layer (capturing deeper interactions)

Dense(1, activation='linear') → Output layer predicts a single continuous value

adam optimizer → Adjusts weights efficiently

mean_squared_error loss → Common for regression problems (predicting numerical values)

3️⃣ Train the Model
(Similar to Coffee Model - Fit with Data)

python
Copy
Edit
# Train the model
model.fit(X_scaled, y_scaled, epochs=500, batch_size=16, verbose=1)
🔹 Explanation

epochs=500 → Trains the model 500 times on the dataset

batch_size=16 → Processes 16 movies at a time

verbose=1 → Shows training progress

4️⃣ Evaluate Model & Make Predictions
(Similar to Coffee Model - Test & Predict)

python
Copy
Edit
# Predict on new movies
test_movie = np.array([[100, 2000, 80, 90, 75, 85]])  # Example test input
test_movie_scaled = scaler_X.transform(test_movie)

predicted_bo_scaled = model.predict(test_movie_scaled)
predicted_bo = scaler_y.inverse_transform(predicted_bo_scaled)

print("Predicted Box Office Collection:", predicted_bo[0][0], "Crores")
🔹 Explanation

Transforms a new movie’s features into scaled values

Model predicts box office collection

Inverse transform to get actual rupees (since we scaled earlier)

Final Summary (Comparing with Coffee Model)
Step	Coffee Roaster Model	Bollywood BO Model
Data Preprocessing	Normalize temperature, time, etc.	Normalize budget, actor popularity, etc.
Input Layer	Features of roasting (temperature, time, etc.)	Features of movies (budget, music, etc.)
Hidden Layers	1 hidden layer	2 hidden layers (more complex interactions)
Output Layer	Roast Quality (classification)	Box Office Collection (regression)
Loss Function	Binary Cross Entropy (for categories)	Mean Squared Error (for numerical values)
Optimizer	Adam Optimizer	Adam Optimizer
Prediction	Predicts roast quality	Predicts movie box office earnings
Next Steps
✅ Train this model on your dataset
✅ Test with multiple movies
✅ If accuracy is low, tweak layers, neurons, and epochs

