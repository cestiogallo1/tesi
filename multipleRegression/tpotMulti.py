import pandas as pd
from tpot import TPOTRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

def main():
    # Caricamento dei dati
    X_train = pd.read_csv("./Dataset/XTrainMultiple.csv")
    y_train = pd.read_csv("./Dataset/YTrainMultiple.csv")
    X_test = pd.read_csv("./Dataset/XTestMultiple.csv")
    y_test = pd.read_csv("./Dataset/YTestMultiple.csv")

    # Inizializza TPOT e avvolgilo in MultiOutputRegressor per la multiregressione
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, scoring='neg_mean_absolute_error', random_state=42)
    multi_output_tpot = MultiOutputRegressor(tpot)

    # Esegui TPOT sulla training set
    multi_output_tpot.fit(X_train, y_train)

    # Valuta il modello ottimale sulla test set
    score = multi_output_tpot.score(X_test, y_test)
    print(f"Score on test set: {score}")

    # Esporta la pipeline ottimizzata (questa esportazione sarà per ogni singola output)
    tpot.export('tpot_best_multi_pipeline.py')

if __name__ == '__main__':
    main()