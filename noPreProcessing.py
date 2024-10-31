# Script per il preprocessing sul dataset

import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
import smogn

#Funzione per togliere i caratteri inutili e convertire in numeri
def clean_column(col):
        if col.dtype == 'object':
            col = col.replace({'$': '', ',': '', '€': '', '%': ''}, regex=True)
            col = pd.to_numeric(col, errors='coerce')
            col = col.str.lower() if col.dtype == 'object' else col
        return col
    
#Pulizia generale del DataFrame 
def clean_df(df):
    # Applicare la funzione di pulizia per rimuovere caratteri inutili e convertire numeri
    df = df.apply(clean_column, axis=0)
    
    # Convertire le colonne di tipo object (stringhe) in datetime, se possibile
    df = df.apply(lambda col: pd.to_datetime(col, errors='coerce') if col.dtype == 'object' else col)
    
    # Eliminare colonne interamente vuote
    df = df.dropna(axis=1, how='all')
    
    # Eliminare colonne con una sola categoria
    df = df.loc[:, df.nunique() > 1]
    
    # Label Encoding per le variabili categoriche
    '''label_encoders = {}  # Dizionario per salvare i LabelEncoders
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le 
    

    return df, label_encoders'''
    return df
    
#Elimino tutte le righe che non hanno valori di paydex    
def deleteEmptyPayDex(df):
    # Rimuove le righe in cui 'PayDexMax15' ha valori NaN
    df = df[~df['PayDexMax15'].isna()]
    return df

#Riempo gli spazi vuoti con KNN
def imputer(df):
    imputer = KNNImputer(n_neighbors=4)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed
    
#Smoter per fare over/under sampling
def smoter(df):
    df_resampled = smogn.smoter(
    df, 
    y='PayDexMax15', 
    )
    return df_resampled

#MutualInformation
def select_features_mutual_information(X, y, threshold=0.01):
    mi_scores = mutual_info_regression(X, y)
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores}) 
    selected_features = feature_scores[feature_scores['Mutual Information'] > threshold]['Feature']
    X_filtered = X[selected_features]
    feature_scores = feature_scores.sort_values(by='Mutual Information', ascending=False)
    
    return X_filtered, feature_scores

def select_features_correlation(X):
    corr_matrix = X.corr().abs()  # Matrice di correlazione assoluta
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]  
    X_selected = X.drop(columns=to_drop)
    return X_selected

#Scaling con MinMaxNormalization    
def scaler(X_train, X_test):
    scaler = MinMaxScaler()
    xd = scaler.fit_transform(X_train)
    xd1 = scaler.transform(X_test)
    X_train = pd.DataFrame(xd, columns=X_train.columns)
    X_test = pd.DataFrame(xd1, columns=X_test.columns)
    return X_train, X_test

def lastPayDex(df):
    # Identificare le colonne che contengono "PayDexMax" nel nome
    paydex_max_columns = [col for col in df.columns if 'PayDexMax' in col]

    # Troviamo l'indice del primo PayDexMax non vuoto per ogni riga
    def primo_paydex_non_vuoto(row):
        for col in paydex_max_columns:
            if not pd.isna(row[col]):
                return col  # Restituisce il nome della prima colonna non vuota
        return None

    df['PrimoPayDexNonVuoto'] = df.apply(primo_paydex_non_vuoto, axis=1)
    
    # Aggiornare la colonna 'PrimoPayDexNonVuoto' con i valori corrispondenti
    def aggiorna_primo_paydex(row):
        primo_paydex = row['PrimoPayDexNonVuoto']
        if primo_paydex is not None and primo_paydex in paydex_max_columns:
            return row[primo_paydex]
        return np.nan

    df['PrimoPayDexNonVuoto'] = df.apply(aggiorna_primo_paydex, axis=1)
    
    return df

# Funzione per trovare gli ultimi 5 valori a partire dall'ultimo PayDex non vuoto
def ultimi_5_paydex(row):
    # Filtra le colonne che contengono 'PayDex'
    paydex_columns = row.filter(like='PayDex')
    
    # Trova l'ultimo valore non vuoto
    last_valid_value = paydex_columns.dropna().iloc[-1] if not paydex_columns.dropna().empty else None
    last_valid_index = paydex_columns.last_valid_index()
    
    if last_valid_index:
        # Trova la posizione dell'ultimo valore non vuoto
        last_index = paydex_columns.index.get_loc(last_valid_index)
        # Prendi gli ultimi 5 valori a partire da quella posizione
        values = paydex_columns.iloc[max(0, last_index-10):last_index+1].values
    else:
        # Se non ci sono valori validi, prendi gli ultimi 5 valori (inclusi NaN)
        values = paydex_columns.iloc[-11:].values

    # Ritorna sempre una lista di 5 valori (anche se ci sono meno di 5)
    return pd.Series(values[::-1].tolist() + [None]*(11 - len(values)))

def rimuovi_paydex_min_max(df):
    columns_to_remove = df.filter(regex='PayDexMin|PayDexMax|PayDex1').columns
    df = df.drop(columns=columns_to_remove)
    return df


    
def main():
    df = pd.read_excel("./Dataset/NETS_Sample2015.xlsx")

    #df[['LastPayDex', 'PayDex1', 'PayDex2', 'PayDex3', 'PayDex4', 'PayDex5', 'PayDex6', 'PayDex7', 'PayDex8', 'PayDex9', 'PayDex10']] = df.apply(ultimi_5_paydex, axis=1)
    
    df = clean_df(df)
    #df = rimuovi_paydex_min_max(df)
    #df = removeOutliers(df)
    df = deleteEmptyPayDex(df)
    df = imputer(df)
    
    #y = df.pop('PayDexMax15')
    columns_to_remove = df.filter(regex='PayDexMin15').columns
    df = df.drop(columns=columns_to_remove)
    
    y = df.pop('PayDexMax15')
    X = df
    
    #FeatureSelection prima per MutualInformation e poi per correlation
    #X, fs = select_features_mutual_information(X,y,threshold=0.01)
    #X = select_features_correlation(X)
    
    print("Creazione e salvataggio training e test set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    #X_train, X_test = scaler(X_train, X_test)

    X_train.to_csv("./Dataset/XTrainN.csv", index=False)
    y_train.to_csv("./Dataset/YTrainN.csv", index=False)
    X_test.to_csv("./Dataset/XTestN.csv", index=False)
    y_test.to_csv("./Dataset/YTestN.csv", index=False)
    
    df = pd.read_csv("./Dataset/YTrainN.csv")

if __name__ == '__main__':
    main()