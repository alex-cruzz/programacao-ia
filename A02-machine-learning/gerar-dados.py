#Importando as libs
import pandas as pd
import numpy as np

#Criando número aleatórios
#Definir uma semente para fins de simulação
np.random.seed(42)

#Gerar 500 registros
n_registros = 500

#Estruturando os dados do arquivo .csv
data = {
    'tempo_contrato': np.random.randint(1,48,n_registros), #1 a 48 meses
    'valor_mensal': np.random.uniform(50.0,150.0,n_registros).round(2), #assinatura com valores que variam de 50 a 150 dinheiros
    'reclamacoes': np.random.poisson(1.5,n_registros)#cada user tem uma média de 1.5 reclamações
}

#Convertendo a estrutura de dicionário em um conjunto de dados
df = pd.DataFrame(data)

#Criar a simulação da lógica churn
#O cliente tem mais chance de sair se tiver muitas reclamações ou se o contrato for curto
df['cancelou']=((df['reclamacoes']>2)|(df['tempo_contrato']<6)).astype(int)

#salvando o dataset em .csv
df.to_csv('churn-dat.csv', index = False)
print("Arquivo 'churn_data.csv' gerado com sucesso!")