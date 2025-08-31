from src.machine_learning.dateset_manipulation import get_dataframe_leituras_sensores
from src.database.tipos_base.database import Database
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_test_split_scaled():
    """
    Realiza o split dos dados em treino e teste, aplicando o MinMaxScaler.
    :return: Tupla com os dados de treino e teste (X_train, X_test, y_train, y_test).
    """

    df = get_dataframe_leituras_sensores()

    features = df.drop(columns=['Manutencao', 'data_leitura'])
    target = df['Manutencao']

    scaler = MinMaxScaler()

    features_scaled = scaler.fit_transform(features)

    return train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

def discover_hyperparameters():
    """
    Função para descobrir os melhores hiperparâmetros para o modelo.
    :return: None
    """
    pass

if __name__ == '__main__':

    pd.set_option('display.max_columns', 50)

    Database.init_sqlite(r'C:\Users\Lucas\PycharmProjects\fiap_sprint3_reply_leo\database.db')
    Database.create_all_tables()

    X_train, X_test, y_train, y_test = train_test_split_scaled()