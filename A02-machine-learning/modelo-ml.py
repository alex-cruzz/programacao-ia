#Etapa 1
#Importando os módulos necessários
import pandas as pd #Ferramenta para criar e alterar dados em tabelas
import numpy as np #Ferramenta de análise matemática

from sklearn.preprocessing import StandardScaler #Padroniza as escalas dos números
from sklearn.ensemble import RandomForestClassifier #Algoritimo de "Florestas Aleatórias"
from sklearn.metrics import classification_report, confusion_matrix #Cria um relatório de classificação; Gera um gráfico para saber se o modelo esta assertivo ou não
from sklearn.model_selection import train_test_split #Permite dividir o algoritimo em teste e treino

import seaborn as sns
import matplotlib.pyplot as pyplot
import joblib

#Etapa 2
try:
    print("Carregando arquivo 'churn-data.csv'...")
    df = pd.read_csv('churn-dat.csv') #Ler o arquivo e criar uma tabela
    print(f"Sucesso, {len(df)} linhas importadas.")
    
except FileNotFoundError:
    print("O arquivo não pode ser encontrado na pasta.")
    exit()    

#Etapa 3 
#Pré processamento de dados (preparar a IA para ser treinada)
#Passo 1: Separar perguntas (x) das respostas (y)
#(x) -> É tudo menos a coluna cancelou, são as "pistas" pro modelo
X = df.drop('cancelou', axis = 1)
#(y) -> apenas a coluna 'cancelou', é o que queremos que o modelo preveja
y = df['cancelou']

#Passo 2: Dividir o treino do teste
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
#test_size = 0.2 separa 20% da massa de dados para testar o modelo

#Passo 3: normalizado (colocando tudo na mesma escala)
scaler = StandardScaler()

#Fit Transform do treino: IA calcula a média e desvio padrão
X_train_scaled = scaler.fit_transform(X_train)

#Fit Transform no teste: Usamos a régua calculada no treino
X_test_scaled = scaler.transform(X_test)

#Etapa 4: Treinar o modelo e realizar a previsão de dados
#Criando o modelo
#n_estimators = 100, cria 100 árvores de decisão 
modelo_churn = RandomForestClassifier(n_estimators = 100, random_state = 42)

#Treinar/ajustar a IA
modelo_churn.fit(X_train_scaled, y_train)

#Prever as respostas
previsoes = modelo_churn.predict(X_test_scaled)

#Etapa 5: Avaliação do modelo
print("Relatório de performance")
print(classification_report(y_test, previsoes))

#Etapa 6: Deploy -> Salvar o trabalho
joblib.dump(modelo_churn,'modelo_churn_v1.pkl')

joblib.dump(scaler,'padronizador_v1.pkl')
print("Arquivos de ML foram exportados com sucesso!")