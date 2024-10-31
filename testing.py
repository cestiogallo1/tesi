# Script che esegue random forest con gli hyperparametri ottenuti e fare testing

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def main():
    X_train = pd.read_csv("./Dataset/XTrain.csv")
    y_train = pd.read_csv("./Dataset/YTrain.csv")
    X_test = pd.read_csv("./Dataset/XTest.csv")
    y_test = pd.read_csv("./Dataset/YTest.csv")
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()    
    
    model = xgb.XGBRegressor(subsample=0.7, reg_alpha=1.0, n_estimators= 200, min_child_weight= 10, max_depth= 15, learning_rate= 0.05, gamma=0.0, colsample_bytree= 0.8)
    #model = RandomForestRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=7, min_samples_split=5, n_estimators=100)
    #model = RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_leaf=2, min_samples_split=5, n_estimators=300, max_depth= None)
    #model = ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=3, min_samples_split=15, n_estimators=100)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(mse)
    print(mae)
    print(r2)
    
    if y_test.shape != y_pred.shape:
        y_pred = y_pred.flatten()  # Appiattisce y_pred se necessario
    
    errors = y_test - y_pred
    mae_per_row = np.abs(errors)  # Errore assoluto per ogni previsione
    results = pd.DataFrame({'Real': y_test, 'Predicted': y_pred, 'Error': mae, 'Absolute Error': mae_per_row})
    print("Errori e residui:\n", results)
    
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    print("\nImportanza delle feature:\n", importance_df)

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")


    # 4. Confronto Errori su Dati Simili
    # Raggruppiamo per una feature importante (es. 'settore') e calcoliamo MAE medio per ogni gruppo
    X_test['Absolute Error'] = mae_per_row  # Aggiungiamo errore assoluto al dataset per l'analisi
    grouped_mae = X_test.groupby('IndustryGroup')['Absolute Error'].mean().sort_values(ascending=False)

    print("\nErrore medio assoluto per settore (ordinato):\n", grouped_mae)
        

    # 5. Plot dei Residui
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Valori previsti")
    plt.ylabel("Residui")
    plt.title("Grafico dei Residui")
    plt.show()
    plt.close()

    
if __name__ == '__main__':
    main()