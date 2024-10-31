import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split


def main():
    X_train = pd.read_csv("./Dataset/XTrainMultiple.csv")
    y_train = pd.read_csv("./Dataset/YTrainMultiple.csv")
    X_test = pd.read_csv("./Dataset/XTestMultiple.csv")
    y_test = pd.read_csv("./Dataset/YTestMultiple.csv")

    #y_train = y_train.values.ravel()
    
    # Inizializza TPOT per la regressione
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, scoring='neg_mean_absolute_error', random_state=42)

    # Esegui TPOT sulla training set
    tpot.fit(X_train, y_train)

    # Valuta il modello ottimale sulla test set
    print(f"Score on test set: {tpot.score(X_test, y_test)}")

    # Esporta la pipeline ottimizzata
    tpot.export('tpot_best_pipeline.py')

if __name__ == '__main__':
    main()