# main.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

print("Iniciando o processo de treinamento dos modelos...")

# --- Funções de Processamento e Treinamento ---

def carregar_e_preparar_dados(filepath='data/df_t.csv'):
    """Carrega e prepara os dados iniciais."""
    print("Carregando dados...")
    df = pd.read_csv(filepath)
    df['datetime_purchase'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])
    df['route'] = df['place_origin_departure'] + ' -> ' + df['place_destination_departure']
    return df

def treinar_modelo_segmentacao(df, k=4):
    """Treina o modelo K-Means e gera o arquivo de clientes segmentados."""
    print("Iniciando treinamento do modelo de segmentação (K-Means)...")
    snapshot_date = df['datetime_purchase'].max() + pd.Timedelta(days=1)
    
    rfm_data = df.groupby('fk_contact').agg({
        'datetime_purchase': lambda date: (snapshot_date - date.max()).days,
        'nk_ota_localizer_id': 'count',
        'gmv_success': 'sum'
    }).reset_index()
    
    rfm_data.rename(columns={'datetime_purchase': 'Recency', 'nk_ota_localizer_id': 'Frequency', 'gmv_success': 'MonetaryValue'}, inplace=True)
    
    rfm_for_scaling = rfm_data[['Recency', 'Frequency', 'MonetaryValue']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_for_scaling)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Salva o resultado para ser usado pelo dashboard
    rfm_data.to_csv('data/clientes_segmentados_rfm.csv', index=False)
    print("Modelo de segmentação treinado e arquivo 'clientes_segmentados_rfm.csv' salvo em data/")
    return rfm_data

def treinar_modelo_recompra(df):
    """Treina e salva o modelo de previsão de recompra."""
    print("Iniciando treinamento do modelo de recompra (XGBoost)...")
    cutoff_date = df['datetime_purchase'].max() - pd.Timedelta(days=30)
    df_train = df[df['datetime_purchase'] < cutoff_date]
    df_target = df[df['datetime_purchase'] >= cutoff_date]
    snapshot_date_model = cutoff_date + pd.Timedelta(days=1)
    
    features_df = df_train.groupby('fk_contact').agg(
        Recency=('datetime_purchase', lambda date: (snapshot_date_model - date.max()).days),
        Frequency=('datetime_purchase', 'count'),
        MonetaryValue=('gmv_success', 'sum')
    ).reset_index()

    target_customers = df_target['fk_contact'].unique()
    features_df['will_buy_in_30_days'] = features_df['fk_contact'].isin(target_customers).astype(int)
    
    X = features_df[['Recency', 'Frequency', 'MonetaryValue']]
    y = features_df['will_buy_in_30_days']
    
    count_majority = y.value_counts()[0]
    count_minority = y.value_counts()[1]
    scale_pos_weight = count_majority / count_minority
    
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight, use_label_encoder=False, 
        eval_metric='logloss', random_state=42
    )
    model.fit(X, y)
    
    # Salva o modelo
    joblib.dump(model, 'models/modelo_recompra.joblib')
    print("Modelo de recompra salvo em models/")
    return model

def treinar_modelo_proxima_rota(df, n_top_routes=50):
    """Treina e salva o modelo de previsão de próxima rota."""
    print("Iniciando treinamento do modelo de próxima rota (RandomForest)...")
    top_n_routes = df['route'].value_counts().nlargest(n_top_routes).index.tolist()
    df['target_route'] = df['route'].apply(lambda x: x if x in top_n_routes else 'Outra')
    df = df.sort_values(by=['fk_contact', 'datetime_purchase'])
    df['last_route'] = df.groupby('fk_contact')['target_route'].shift(1)
    df_predict = df.dropna(subset=['last_route'])
    
    X = df_predict[['last_route']]
    y = df_predict['target_route']
    
    le = LabelEncoder()
    X['last_route_encoded'] = le.fit_transform(X['last_route'])
    X = X.drop('last_route', axis=1)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    
    # Salva o modelo e o encoder
    joblib.dump(model, 'models/modelo_proxima_rota.joblib')
    joblib.dump(le, 'models/encoder_proxima_rota.joblib')
    print("Modelo de próxima rota e encoder salvos em models/")
    return model, le


if __name__ == "__main__":
    # Orquestra a execução de todas as etapas
    df_principal = carregar_e_preparar_dados(filepath='data/df_amostra.csv')
    treinar_modelo_segmentacao(df_principal)
    treinar_modelo_recompra(df_principal)
    treinar_modelo_proxima_rota(df_principal)
    print("\nProcesso de treinamento concluído com sucesso!")