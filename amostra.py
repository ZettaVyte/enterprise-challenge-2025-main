import pandas as pd

# Carrega o seu arquivo de dados completo
df_completo = pd.read_csv('data/df_t.csv')

# Cria uma amostra com as primeiras 10.000 linhas
df_amostra = df_completo.head(10000)

# Salva a nova amostra em um arquivo separado
df_amostra.to_csv('data/df_amostra.csv', index=False)

print("Arquivo 'df_amostra.csv' criado com sucesso na pasta data/!")