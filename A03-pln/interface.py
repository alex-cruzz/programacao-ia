import streamlit as st 
import joblib
import spacy
import pandas as pd

#Configuração da página (título e ícone)
st.set_page_config(page_title = "Triagem de chamados", page_icon = "")

#Carregamento de recursos
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_triagem_suporte.pkl") # Corrigido: return na mesma linha

@st.cache_resource
def carregar_nlp():
    return spacy.load("pt_core_news_sm") # Corrigido: return na mesma linha

try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()
except:
    st.error("Erro: Execute o script 'treinar_modelo_py' para gerar o arquivo .pkl")
    st.stop()
    
#Lógica de processamento
def analisar_chamado(texto_usuario):
    doc = nlp(texto_usuario)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]

    texto_limpo = " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
    ])
    
    categoria_predita = modelo.predict([texto_limpo])[0] 
    probs = modelo.predict_proba([texto_limpo])[0]
    
    # Corrigido: 'probs' em vez de 'proba' para bater com a variável acima
    confianca = max(probs)*100
    
    return categoria_predita, confianca, entidades
    
#------Interface Gráfica------
st.title("Triagem de suporte") 
st.markdown("Descreva o problema em poucas palavras.") 

if "messages" not in st.session_state:
    st.session_state.messages = [] 
    
for message in st.session_state.messages: 
    with st.chat_message(message["role"]): 
        st.markdown(message["content"]) 

if prompt:= st.chat_input("Ex.: O servidor AWS parou de responder..."):
    
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({"role":"user", "content": prompt})
    
    # CORREÇÃO: O processamento deve estar dentro deste 'if' (identado)
    categoria, confianca, ents = analisar_chamado(prompt)

    resposta_md = f"""
**Análise do chamado:**
**Categoria:** `{categoria}`
**Confiança:** `{confianca:.2f}%`
"""
    
    if ents:
        resposta_md +="\n\n **Entidades detectadas:**"
        for ent in ents:
            # Corrigido: ent[0] para o texto e ent[1] para o rótulo
            resposta_md += f"\n- {ent[0]} *({ent[1]})*"

    #Ações automáticas por categoria  
    acoes = {
        "Infraestrutura": "Encaminhamento para equipe N2",
        "Acesso": "Verificando logs de autenticação",
        "Hardware": "Abrindo ordem de serviço.",
        "Software": "Verificando disponibilidade de licenças"
    }
        
    #Adicionar as ações sugeridas com base na categoria
    resposta_md += f"\n\n **Ação:** {acoes.get(categoria,'Triagem manual necessária.')}"
    
    #3. Exibir a resposta do assistente
    with st.chat_message('assistant'):
        st.markdown(resposta_md)
        
    #Salvar resposta no histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta_md 
    })