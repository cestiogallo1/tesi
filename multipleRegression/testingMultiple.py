# Script che esegue random forest con gli hyperparametri ottenuti e fare testing

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb




def main():
    X_train = pd.read_csv("./Dataset/Multiple/XTrainMultiple.csv")
    y_train = pd.read_csv("./Dataset/Multiple/YTrainMultiple.csv")
    X_test = pd.read_csv("./Dataset/Multiple/XTestMultiple.csv")
    y_test = pd.read_csv("./Dataset/Multiple/YTestMultiple.csv")
    
    #y_train = y_train.values.ravel()
    

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

    
if __name__ == '__main__':
    main()