#Import da bibliotecas necessárias
import pandas as pd #Manipulação de dados em forma de tabela
import spacy #Lib de processamento de linguagem natural
import joblib #Salvar e carregar modelos de ia treinados
from sklearn.feature_extraction.text import TfidfVectorizer #Converte texto em vetores
from sklearn.naive_bayes import MultinomialNB #Classifica texto com base em porbabilidade
from sklearn.pipeline import make_pipeline #Junta várias etapas num fluxo só
from sklearn.model_selection import train_test_split #Divide o conjunto de dados em treino e teste
from sklearn.metrics import classification_report #Avalia o modelo

#Etapa 1: Carregar dados
print("Carregando dataset...")
df = pd.read_csv("dataset_chamados.csv")

#Etapa 2: Pipeline de processamento focado em performance
#Vamos usar o Spacy dentro do fluxo da UI
nlp = spacy.load("pt_core_news_sm") #Carregamento da lib da spay em português

def prep(texto):
    doc = nlp(texto) #Processamento do texto (tokenização e análise probabilística)
    
    return " ".join({
        token.lemma_.lower()
        for token in doc
        if not token.is_punct #Remove qualquer tipo de pontuação
    })
print("Processando textos, pode levar alguns instantes.")

df['texto_limpo'] = df['texto'].apply(prep) #Aplicar a função de limpeza na col de texto 

#Etapa 3: Dividir entre treino e teste
#X = textos de entrada
#y = categorias (labels)
X_train, X_test, y_train, y_test = train_test_split(
    df["texto_limpo"], #Dados de entrada com pré processsamento
    df["categoria"],   #Categorias
    test_size = 0.2      #20% pra teste
)

#Etapa 4: Criar e treinar pipeline de ML
model_pipeline = make_pipeline(
    TfidfVectorizer(), #Converter texto em valor numérico
    MultinomialNB()      #Aplica classificador Naive Bayes (palavra : intenção/categoria)
)

#Treina modelo com os dados de treino
model_pipeline.fit(X_train, y_train)      

#Etapa 5: Salvar modelo treinado
joblib.dump(model_pipeline, "modelo_triagem_suporte.pkl")
print("Modelo treinado e salvo como modelo_triagem_suporte.pkl")

