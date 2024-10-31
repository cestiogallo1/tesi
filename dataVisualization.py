from matplotlib import pyplot as plt
import pandas as pd
from preProcessing import deleteEmptyPayDex, lastPayDex, clean_df
import seaborn as sns

def main():
    df = pd.read_excel("./Dataset/NETS_Sample2015.xlsx")
    #df = deleteEmptyPayDex(df)
    df = lastPayDex(df)
    
    # Raggruppa per 'IndustryGroup' e calcola la media del PayDex
    industry_paydex_mean = df.groupby('IndustryGroup')['PrimoPayDexNonVuoto'].mean().reset_index()

    # Ordina in base al PayDex medio e prendi le prime 20 industrie
    industry_paydex_mean = industry_paydex_mean.sort_values(by='PrimoPayDexNonVuoto', ascending=False).head(5)

    # Creazione del grafico a barre per visualizzare il PayDex medio per le prime 20 industrie
    plt.figure(figsize=(15, 8))
    bar_plot = sns.barplot(x='IndustryGroup', y='PrimoPayDexNonVuoto', data=industry_paydex_mean, width=0.7)


    # Aggiungi titolo e etichette
    plt.title('Top industrie per PayDex Medio', fontsize=16)
    plt.xlabel('Industry', fontsize=20)
    plt.ylabel('PayDex Medio', fontsize=12)
    plt.ylim(80, 90)
    plt.xticks(rotation=45, ha='right', fontsize = 20)

    
    for p in bar_plot.patches:
    # Ottieni le coordinate del centro della barra
        plt.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.5, f'{p.get_height():.2f}', 
                ha='center', va='bottom', fontsize=14)  # `va='bottom'` per posizionare il testo sopra la barra

    # Filtra i dati per rimuovere righe con PayDex non valido (null o pari a 0)
    df_filtered = df[df['PrimoPayDexNonVuoto'] > 0]

    # Raggruppa per 'IndustryGroup' e calcola la media del PayDex
    industry_paydex_mean = df_filtered.groupby('IndustryGroup')['PrimoPayDexNonVuoto'].mean().reset_index()

    # Ordina in base al PayDex medio in modo crescente e prendi le peggiori 20 industrie
    industry_paydex_mean = industry_paydex_mean.sort_values(by='PrimoPayDexNonVuoto', ascending=True).head(5)

    # Creazione del grafico a barre per visualizzare il PayDex medio per le peggiori 20 industrie
    plt.figure(figsize=(15, 8))
    bar_plot = sns.barplot(x='IndustryGroup', y='PrimoPayDexNonVuoto', data=industry_paydex_mean)

    # Aggiungi titolo e etichette
    plt.title('Peggiori industrie per PayDex Medio', fontsize=16)
    plt.xlabel('Industry', fontsize=12)
    plt.ylabel('PayDex Medio', fontsize=12)


    # Ruota le etichette dell'asse x per renderle più leggibili
    plt.xticks(rotation=45, ha='right', fontsize = 20)
    for p in bar_plot.patches:
    # Ottieni le coordinate del centro della barra
        plt.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.5, f'{p.get_height():.2f}', 
                ha='center', va='bottom', fontsize=14)  # `va='bottom'` per posizionare il testo sopra la barra

    # Calcola la distribuzione delle aziende per industria
    industry_counts = df_filtered['IndustryGroup'].value_counts()

    # Calcola le percentuali
    industry_percentages = industry_counts / industry_counts.sum() * 100

    # Raggruppa le categorie sotto l'1% in "Altro"
    other_count = industry_counts[industry_percentages < 1].sum()  # Somma le categorie < 1%
    industry_percentages = industry_percentages[industry_percentages >= 1]  # Mantieni solo le categorie >= 1%

    # Aggiungi la categoria "Altro" se ci sono categorie raggruppate
    if other_count > 0:
        # Crea una nuova serie per "Altro"
        other_series = pd.Series({'Altro': other_count})
        industry_percentages = pd.concat([industry_percentages, other_series])

    # Creazione del grafico a torta
    plt.figure(figsize=(10, 10))
    plt.pie(industry_percentages, labels=industry_percentages.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuzione delle Aziende per Tipologia di Industria', fontsize=16)
    plt.axis('equal')  # Per assicurare che il grafico sia un cerchio
    
    # Mostra il grafico
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()