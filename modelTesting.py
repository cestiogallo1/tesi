# Script per comparazione modelli con i parametri di base

from numpy import mean
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_validate

scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error)
}

models = [
    RandomForestRegressor(),
    LinearRegression(),
    AdaBoostRegressor(n_estimators=50, random_state=42),
    Ridge(alpha=1.0),
    Lasso(alpha=1.0),
    SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    KNeighborsRegressor(n_neighbors=5),
    xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
]

def modelTesting(X_train, y_train):
    for model in models:
        results = cross_validate(model, X_train, y_train, cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=42), n_jobs=8, scoring=scoring_metrics)
        mean_r2 = mean(results['test_r2'])
        mean_mse = mean(results['test_mse'])
        mean_mae = mean(results['test_mae'])

        # Stampa le medie
        print(str(model))
        print(f"Media R^2: {mean_r2:.4f}")
        print(f"Media MSE: {mean_mse:.4f}")
        print(f"Media MAE: {mean_mae:.4f}")
    


def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")

    y_train = y_train.values.ravel()
    
    modelTesting(X_train, y_train)


if __name__ == '__main__':
    main()