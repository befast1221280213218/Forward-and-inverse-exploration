import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

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

# Combine into a two-column matrix
data_normalized = np.concatenate((data_A, data_B, data_C, data_D, data_E, data_F, data_G, data_H), axis=1)

# Split into input (columns EFGH) and output (columns ABCD)
X = data_normalized[:, 4:]  # EFGH columns as input
y = data_normalized[:, :4]  # ABCD columns as output
r2_best = 0

# Using the best hyperparameters to train the final model
rf_regressor = RandomForestRegressor(n_estimators=best_params['n_estimators'], 
                                     max_depth=best_params['max_depth'], 
                                     min_samples_split=best_params['min_samples_split'], 
                                     random_state=i_best)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)
final_r2 = r2_score(y_test, y_pred_rf)

print(f"Final Random Forest Regression (after tuning) - Best R^2: with test split ratio {j_best} and random_state: {i_best}")

# Using the best hyperparameters to train the final model
final_rf_regressor = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           min_samples_split=best_params['min_samples_split'],
                                           random_state=i_best)

# Calculate MAE, RMSE, and R2 for each column
mae_list = []
rmse_list = []
r2_list = []
for i in range(4):
    y_pred_i = y_pred_rf[:, i]
    y_test_i = y_test[:, i]
    mae_i = mean_absolute_error(y_test_i, y_pred_i)
    rmse_i = math.sqrt(mean_squared_error(y_test_i, y_pred_i))
    r2_i = r2_score(y_test_i, y_pred_i)
    mae_list.append(mae_i)
    rmse_list.append(rmse_i)
    r2_list.append(r2_i)

for i in range(4):
    print(f"MAE for column {i}: {mae_list[i]}")
    print(f"RMSE for column {i}: {rmse_list[i]}")
    print(f"R2 for column {i}: {r2_list[i]}")
