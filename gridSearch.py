from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import xgboost as xgb

scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error)
}


def training(nome, model, params, X_train, X_test, y_test, y_train):
    
    grid_search = GridSearchCV(
    estimator=model, 
    param_grid=params, 
    cv=5, 
    verbose=3, 
    refit='r2',
    n_jobs=-1,
    scoring=scoring_metrics
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found for {nome}: ", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{nome} Mse: {mse:.4f}")
    print(f"{nome} Mae: {mae:.4f}")
    with open('grid_search_results.txt', 'w') as f:
        f.write(f"{nome} Mse: {mse:.4f}")
        f.write(f"{nome} Mae: {mae:.4f}")


def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    X_test = pd.read_csv("./Dataset/XTest.csv")
    y_test = pd.read_csv("./Dataset/YTest.csv")
    y_train = y_train.values.ravel()
    
    model = RandomForestRegressor()
    
    params = {
    'n_estimators': np.arange(200, 400, 10),
    'max_features': ['sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split':  np.arange(2, 10),
    'min_samples_leaf': np.arange(1, 8),
    'bootstrap': [True, False]
    }
    
    training("Random Forest", model, params, X_train, X_test, y_test, y_train)
    
    
    model = xgb.XGBRegressor()
    params = { 
    'learning_rate' : [0.01,0.05,0.10,0.15,0.20,0.25,0.30],
    'max_depth' : np.arange(8, 20),
    'min_child_weight' : np.arange(4,15),
    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6 ],
    'n_estimators': np.arange(100, 300, 10),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7, 0.8, 0.9 ]
    }
    
    training("XGB", model, params, X_train, X_test, y_test, y_train)
    

if __name__ == '__main__':
    main()

