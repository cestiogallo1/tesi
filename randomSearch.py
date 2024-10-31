from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error)
}


def training(nome, model, params, X_train, X_test, y_test, y_train):
    
    multi_output_model = MultiOutputRegressor(model)
    
    random_search = RandomizedSearchCV(
        estimator=multi_output_model, 
        param_distributions=params, 
        n_iter=100, 
        cv=5, 
        verbose=3, 
        refit=False,
        random_state=42, 
        n_jobs=-1,
        scoring=scoring_metrics
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters found for {nome}: ", random_search.best_params_)

    y_pred = random_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{nome} Mse: {mse:.4f}")
    print(f"{nome} Mae: {mae:.4f}")


def main():
    X_train = pd.read_csv("./Dataset/XTrainMultiple.csv")
    y_train = pd.read_csv("./Dataset/YTrainMultiple.csv")
    X_test = pd.read_csv("./Dataset/XTestMultiple.csv")
    y_test = pd.read_csv("./Dataset/YTestMultiple.csv")
    
    model = RandomForestRegressor()
    
    params = {
        'estimator__n_estimators': [10, 30, 50, 70, 100, 150, 200, 300],
        'estimator__max_features': ['auto', 'sqrt', 'log2'],
        'estimator__max_depth': [10, 20, 30, None],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'estimator__bootstrap': [True, False]
    }
    
    #training("Random Forest", model, params, X_train, X_test, y_test, y_train)
    
    
    model = xgb.XGBRegressor()
    
    params = { 
        'estimator__learning_rate' : [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'estimator__max_depth' : [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
        'estimator__min_child_weight' : [1, 3, 5, 7, 10],
        'estimator__gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'estimator__n_estimators': [100, 200, 500],
        'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'estimator__reg_alpha': [0, 0.1, 0.5, 1.0],
        'estimator__colsample_bytree' : [0.3, 0.4, 0.5 , 0.7, 0.8, 0.9]
    }
    
    training("XGB", model, params, X_train, X_test, y_test, y_train)

if __name__ == '__main__':
    main()