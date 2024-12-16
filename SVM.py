import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt

data_path = r'./dataset/ETT-small/ETTh2.csv'
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

if not df.index.is_monotonic_increasing:
    df = df.sort_index()

target_col = 'HUFL'
df_target = df[[target_col]]

def create_lagged_features(df, lag=1):
    df_shifted = df.shift(lag)
    df_shifted.columns = [f'{col}_lag{lag}' for col in df.columns]
    return pd.concat([df, df_shifted], axis=1).dropna()

lags_to_try = range(0, 49)  
best_lag = None
best_mse = float('inf')
best_model = None
best_scaler_X = None
best_scaler_y = None
best_X_train_scaled = None
best_y_train_scaled = None
best_X_test_scaled = None
best_y_test_scaled = None

for lag in lags_to_try:
    print(f"Trying lag: {lag}")
    
    df_lagged = create_lagged_features(df_target, lag)

    X = df_lagged.drop(target_col, axis=1)
    y = df_lagged[target_col]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []

    for train_index, val_index in tscv.split(X_train_scaled):
        X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_cv, y_val_cv = y_train_scaled[train_index], y_train_scaled[val_index]

        svr_model.fit(X_train_cv, y_train_cv.ravel())

        y_pred_cv = svr_model.predict(X_val_cv)
        y_pred_cv_unscaled = scaler_y.inverse_transform(y_pred_cv.reshape(-1, 1))
        y_val_cv_unscaled = scaler_y.inverse_transform(y_val_cv)
        mse = mean_squared_error(y_val_cv_unscaled, y_pred_cv_unscaled)
        mse_scores.append(mse)

    avg_mse = np.mean(mse_scores)
    print(f"Lag {lag}: Average MSE = {avg_mse}")

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_lag = lag
        best_model = svr_model
        best_scaler_X = scaler_X
        best_scaler_y = scaler_y
        best_X_train_scaled = X_train_scaled
        best_y_train_scaled = y_train_scaled
        best_X_test_scaled = X_test_scaled
        best_y_test_scaled = y_test_scaled

print(f"Best lag found: {best_lag} with MSE: {best_mse}")

y_pred_final_scaled = best_model.predict(best_X_test_scaled)
y_pred_final = best_scaler_y.inverse_transform(y_pred_final_scaled.reshape(-1, 1))

mse_final = mean_squared_error(y_test, y_pred_final)
print(f'Final Mean Squared Error: {mse_final}')

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual', color='blue', alpha=0.5)
plt.plot(y_test.index, y_pred_final, label='Predicted', color='orange', linestyle='--')
plt.title('Actual vs Predicted Values using SVM with Optimal Lag')
plt.legend()
plt.show()

df_forecast = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_final.flatten()}, index=y_test.index)

df_forecast.to_csv('forecast_results_svm_best_lag.csv')

plt.savefig('forecast_vs_actual_svm_best_lag.png')