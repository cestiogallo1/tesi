import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
import smogn

anni = 12

# Funzione per togliere i caratteri inutili e convertire in numeri
def clean_column(col):
    if col.dtype == 'object':
        col = col.replace({'$': '', ',': '', '€': '', '%': ''}, regex=True)
        col = pd.to_numeric(col, errors='coerce')
        col = col.str.lower() if col.dtype == 'object' else col
    return col

# Pulizia generale del DataFrame 
def clean_df(df):
    df = df.apply(clean_column, axis=0)
    df = df.apply(lambda col: pd.to_datetime(col, errors='coerce') if col.dtype == 'object' else col)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, df.nunique() > 1]
    return df

# Funzione per riempire gli spazi vuoti con KNN
def imputer(df):
    imputer = KNNImputer(n_neighbors=4)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    return df_imputed

# Funzione per smoter per fare over/under sampling
def smoter(df):
    df_resampled = smogn.smoter(df, y='PayDexMax15')
    return df_resampled

# Selezione delle feature tramite Mutual Information
def select_features_mutual_information(X, y, threshold=0.01):
    mi_scores = mutual_info_regression(X, y)
    feature_scores = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores}) 
    selected_features = feature_scores[feature_scores['Mutual Information'] > threshold]['Feature']
    X_filtered = X[selected_features]
    feature_scores = feature_scores.sort_values(by='Mutual Information', ascending=False)
    return X_filtered, feature_scores

# Selezione delle feature tramite correlazione
def select_features_correlation(X):
    corr_matrix = X.corr().abs()  
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]  
    X_selected = X.drop(columns=to_drop)
    return X_selected

# Scaling con MinMaxScaler
def scaler(X_train, X_test):
    scaler = MinMaxScaler()
    xd = scaler.fit_transform(X_train)
    xd1 = scaler.transform(X_test)
    X_train = pd.DataFrame(xd, columns=X_train.columns)
    X_test = pd.DataFrame(xd1, columns=X_test.columns)
    return X_train, X_test

# Funzione per trovare gli ultimi 5 anni di valori a partire dall'ultimo PayDex non vuoto
def ultimi_paydex(row):
    # Filtra le colonne che contengono 'PayDex'
    paydex_columns = row.filter(like='PayDex')
    
    # Trova l'ultimo valore non vuoto
    last_valid_value = paydex_columns.dropna().iloc[-1] if not paydex_columns.dropna().empty else None
    last_valid_index = paydex_columns.last_valid_index()
    
    if last_valid_index:
        # Trova la posizione dell'ultimo valore non vuoto
        last_index = paydex_columns.index.get_loc(last_valid_index)
        # Prendi gli ultimi valori a partire da quella posizione
        values = paydex_columns.iloc[max(0, last_index-(anni*2 + 1)):last_index+1].values
        ultimo_anno = int(last_valid_index.replace('PayDexMax', ''))
    else:
        # Se non ci sono valori validi, prendi gli ultimi 5 valori (inclusi NaN)
        values = paydex_columns.iloc[-(anni * 2 + 2):].values
        ultimo_anno = None
        
    # Ritorna sempre una lista di n valori (anche se ci sono meno di n)
    paydex_series = pd.Series(values[::-1].tolist() + [None] * ((anni * 2 + 2) - len(values)))
    
    # Aggiungi l'anno come ultimo elemento in un'altra Serie
    ultimo_anno_series = pd.Series([ultimo_anno], index=['UltimoAnnoUtile'])

    # Concatenare le due serie
    return pd.concat([paydex_series, ultimo_anno_series])

def ultimi_FIPS(row):
    # Filtra le colonne che contengono 'FIPS'
    columns = row.filter(like="FIPS")
    
    # Trova l'anno dell'ultima colonna 'UltimoAnnoUtile'
    ultimo_anno = row.get('UltimoAnnoUtile', None)
    
    if ultimo_anno is not None and not pd.isna(ultimo_anno):
        # Trova l'indice di FIPS corrispondente all'anno
        fips_index = columns.filter(like=str(int(ultimo_anno))).index
        
        if not fips_index.empty:
            # Trova la posizione dell'ultimo valore non vuoto
            last_index = columns.index.get_loc(fips_index[0])
            # Prendi gli ultimi 'anni' valori a partire da quella posizione
            start_index = max(0, last_index - anni)  
            values = columns.iloc[start_index:last_index].values
        else:
            # Se non ci sono valori validi, riempi con None
            values = [None] * (anni)
    else:
        # Se 'UltimoAnnoUtile' non è disponibile, riempi con None
        values = [None] * (anni)

    # Ritorna sempre una lista con il numero richiesto di valori (anche se ci sono meno di quelli disponibili)
    return pd.Series(list(values) + [None] * ((anni) - len(values)))


def ultimi_DnB(row):
    # Filtra le colonne che contengono 'DnB'
    columns = row.filter(like="DnBRating")
    
    # Trova l'anno dell'ultima colonna 'UltimoAnnoUtile'
    ultimo_anno = row.get('UltimoAnnoUtile', None)
    
    if ultimo_anno is not None and not pd.isna(ultimo_anno):
        # Trova l'indice di  corrispondente all'anno
        fips_index = columns.filter(like=str(int(ultimo_anno))).index
        
        if not fips_index.empty:
            # Trova la posizione dell'ultimo valore non vuoto
            last_index = columns.index.get_loc(fips_index[0])
            # Prendi gli ultimi 'anni' valori a partire da quella posizione
            start_index = max(0, last_index - anni)  
            values = columns.iloc[start_index:last_index].values
        else:
            # Se non ci sono valori validi, riempi con None
            values = [None] * (anni)
    else:
        # Se 'UltimoAnnoUtile' non è disponibile, riempi con None
        values = [None] * (anni)

    # Ritorna sempre una lista con il numero richiesto di valori (anche se ci sono meno di quelli disponibili)
    return pd.Series(list(values) + [None] * ((anni) - len(values)))

def ultimi_Sales(row):
    # Filtra le colonne che contengono 'FIPS'
    columns = row.filter(like="SalesC")
    
    # Trova l'anno dell'ultima colonna 'UltimoAnnoUtile'
    ultimo_anno = row.get('UltimoAnnoUtile', None)
    
    if ultimo_anno is not None and not pd.isna(ultimo_anno):
        # Trova l'indice corrispondente all'anno
        fips_index = columns.filter(like=str(int(ultimo_anno))).index
        
        if not fips_index.empty:
            # Trova la posizione dell'ultimo valore non vuoto
            last_index = columns.index.get_loc(fips_index[0])
            # Prendi gli ultimi 'anni' valori a partire da quella posizione
            start_index = max(0, last_index - anni)  
            values = columns.iloc[start_index:last_index].values
        else:
            # Se non ci sono valori validi, riempi con None
            values = [None] * (anni)
    else:
        # Se 'UltimoAnnoUtile' non è disponibile, riempi con None
        values = [None] * (anni)

    # Ritorna sempre una lista con il numero richiesto di valori (anche se ci sono meno di quelli disponibili)
    return pd.Series(list(values) + [None] * ((anni) - len(values)))




def rimuovi_paydex_min_max(df):
    columns_to_remove = df.filter(regex='PayDexMin|PayDexMax|PayDex1$|PayDex3|PayDex5|UltimoAnnoUtile').columns
    df = df.drop(columns=columns_to_remove)
    return df

def rimuovi_fips(df):
    columns_to_remove = df.filter(regex='FIPS').columns
    df = df.drop(columns=columns_to_remove)
    return df

def rimuovi_dnb(df):
    columns_to_remove = df.filter(regex='DnBRating').columns
    df = df.drop(columns=columns_to_remove)
    return df

def rimuovi_sales(df):
    columns_to_remove = df.filter(regex='SalesC').columns
    df = df.drop(columns=columns_to_remove)
    return df


# Main per gestire la predizione dei prossimi tre PayDex
def main():
    df = pd.read_excel("./Dataset/NETS_Sample2015.xlsx")

    n_paydex = anni * 2 + 1
    
    paydex_columns = [f'LastPayDex'] + [f'PayDex{i}' for i in range(1, n_paydex + 1)] + ['UltimoAnnoUtile']
    df[paydex_columns]= df.apply(ultimi_paydex, axis=1) 
    
    FIPS_columns = [f'F{i}' for i in range(1, anni + 1)]
    df[FIPS_columns] = df.apply(ultimi_FIPS, axis = 1)
    
    DnB_columns = [f'DnB{i}' for i in range(1, anni + 1)]
    df[DnB_columns] = df.apply(ultimi_DnB, axis = 1)
    
    Sales_columns = [f'S{i}' for i in range(1, anni + 1)]
    df[Sales_columns] = df.apply(ultimi_Sales, axis = 1)
    
    # encoding feature categoriche
    o = OrdinalEncoder()
    for index in df.columns:
        if df[index].dtype == 'O':
            df[index] = o.fit_transform(df[index].values.reshape(-1, 1))
    
    df = clean_df(df)
    df = rimuovi_paydex_min_max(df)
    df = rimuovi_fips(df)
    df = rimuovi_dnb(df)
    df = rimuovi_sales(df)
    df = imputer(df)

    # Definizione delle variabili target come LastPayDex, PayDex2, e PayDex4
    y = df[['LastPayDex', 'PayDex2', 'PayDex4']]
    X = df.drop(columns=['LastPayDex', 'PayDex2', 'PayDex4'])

    #X, fs = select_features_mutual_information(X, y, threshold=0.01)
    X = select_features_correlation(X)
    
    print("Creazione e salvataggio training e test set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train, X_test = scaler(X_train, X_test)

    X_train.to_csv("./Dataset/Multiple/XTrainMultiple.csv", index=False)
    y_train.to_csv("./Dataset/Multiple/YTrainMultiple.csv", index=False)
    X_test.to_csv("./Dataset/Multiple/XTestMultiple.csv", index=False)
    y_test.to_csv("./Dataset/Multiple/YTestMultiple.csv", index=False)
    
if __name__ == '__main__':
    main()