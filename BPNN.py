import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import os
from tqdm import tqdm

# Read the Excel file
file_path = r'***.xlsx'
df = pd.read_excel(file_path)

# Select necessary columns
selected_columns_A = ['A']
selected_columns_B = ['B']
selected_columns_C = ['C']
selected_columns_D = ['D']
selected_columns_E = ['E']
selected_columns_F = ['F']
selected_columns_G = ['G']
selected_columns_H = ['H']

# Normalize the columns
A = **  
df['A'] = df['A'] / A
df['B'] = df['B'] / A
df['C'] = df['C'] / A
df['D'] = df['D'] / A
df['E'] = df['E'] / A
df['F'] = df['F'] / A
df['G'] = df['G'] / A
df['H'] = df['H'] / A

# Convert the processed data to NumPy arrays
data_A = df[selected_columns_A].to_numpy()
data_B = df[selected_columns_B].to_numpy()
data_C = df[selected_columns_C].to_numpy()
data_D = df[selected_columns_D].to_numpy()
data_E = df[selected_columns_E].to_numpy()
data_F = df[selected_columns_F].to_numpy()
data_G = df[selected_columns_G].to_numpy()
data_H = df[selected_columns_H].to_numpy()

# Combine into a matrix
data_normalized = np.concatenate((data_A, data_B, data_C, data_D, data_E, data_F, data_G, data_H), axis=1)

# Split into input (columns EFGH) and output (columns ABCD)
X = data_normalized[:, 4:]  # EFGH columns as input
y = data_normalized[:, :4]  # ABCD columns as output

best_r2 = -float('inf')  # Initialize with a very small value
best_rmse = float('inf')  # Initialize with a very large value
best_aic = float('inf')  # Initialize with a very large value
best_mae = float('inf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
results = []
R2 = []

# Try different combinations of hyperparameters
n_trials = 1  # Number of trials (set as per requirement)
progress_bar = tqdm(total=n_trials)

for i in range(B, C):  # Define B and C with appropriate values
    for j in range(10, 11):
        # Split the dataset into training and testing sets
        j = round(j / 100, 2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=j, random_state=i)

        # Reshape data for Conv1D layer (samples, timesteps, features)
        X_train_conv = np.expand_dims(X_train, axis=-1)
        X_test_conv = np.expand_dims(X_test, axis=-1)

        # Build the neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_conv.shape[1], X_train_conv.shape[2])),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4)  # Output has 4 columns corresponding to ABCD
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

        # Train the model
        history = model.fit(X_train_conv, y_train, epochs=D, batch_size=32, validation_data=(X_test_conv, y_test), verbose=0)

        # Evaluate the model
        loss, mae, mse = model.evaluate(X_test_conv, y_test, verbose=0)

        # Make predictions with the model
        y_pred = model.predict(X_test_conv)

        # Calculate R2 score
        r2 = r2_score(y_test, y_pred)
        print(r2)
        R2.append(r2)

        # Calculate metrics for each column
        for column_index in range(y_test.shape[1]):
            column_r2 = r2_score(y_test[:, column_index], y_pred[:, column_index])
            column_mae = mean_absolute_error(y_test[:, column_index], y_pred[:, column_index])
            column_rmse = np.sqrt(mean_squared_error(y_test[:, column_index], y_pred[:, column_index]))

            # Print performance metrics for each column
            print(f"Column {column_index} R2: {column_r2:.5f}, MAE: {column_mae:.5f}, RMSE: {column_rmse:.5f}")
