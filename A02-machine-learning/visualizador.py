#importar módulos
import streamlit as st #lib que transforma python em sites
import joblib #Salva e exporta o modelo treinado de IA em um binário
import numpy as np #lib para organizar os dados numéricos

#Passo 1: Configurando a aba do navegador
st.set_page_config(page_title="Análise de churn",page_icon="🔥")

#Textos da tela principal
st.title("Sistema de retenção de base") #Título da página
st.markdown("Insira dados do cliente para verificar risco de cancelamento")

#Passo 2: Importar os dados da inteligência artificial com  joblib
modelo = joblib.load('modelo_churn_v1.pkl') #Carrega as regras de decisão do modelo
scaler = joblib.load('padronizador_v1.pkl') #Carrega a régua matemática

#Passo 3: Criar a interface de entrada com um formulário
col1, col2 = st.columns(2) #Criando duas colunas

#Coluna lado esquerdo (col1)
with col1:
    tempo = st.number_input("Tempo de contrato (meses)", min_value = 1, value = 12, max_value = 96)
    valor = st.number_input("Valor da assinatura: (R$)", min_value = 0.0, value = 50.0)

with col2:
    reclamacoes = st.slider("Histórico de reclamações", 0, 10, 1)
    
#Passo 4: Processamento de dados
if st.button("Analisar risco"):
    dados = scaler.transform([[tempo,valor,reclamacoes]]) 
    probabilidade = modelo.predict_proba(dados)[0][1]

#Previsão de probabilidade

#Passo 5: Feedback de negócios
    st.divider() #Cria uma linha divisória

#Probabilidade maior que 70%
    if probabilidade > 0.7:
        st.error(f"*ALTO RISCO DE CHURN* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:* Oferecer cupom de fidelidade FID2103") 
    elif probabilidade > 0.3:
        st.warning(f"*Risco moderado de churn* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:* Realizar chamada de acompanhamento.")
    else:
        st.success(f"*Cliente estável* ({probabilidade*100:.1f}%)")        
        st.info("Nada a realizar no momento.")
