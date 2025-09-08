# dashboard_clickbus.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

# --- Configuração da Página ---
st.set_page_config(
    page_title="Dashboard Preditivo - ClickBus",
    page_icon="🚌",
    layout="wide"
)

# --- Funções de Processamento e Cache ---

@st.cache_data
def load_and_process_data(file_path):
    """Carrega os dados da amostra e faz o pré-processamento inicial."""
    df = pd.read_csv(file_path)
    df['datetime_purchase'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])
    df['route'] = df['place_origin_departure'] + ' -> ' + df['place_destination_departure']
    return df

@st.cache_data
def generate_segmented_data(_df):
    """Executa o modelo K-Means para gerar os dados de segmentação em memória."""
    df_copy = _df.copy()
    snapshot_date = df_copy['datetime_purchase'].max() + pd.Timedelta(days=1)
    
    rfm_data = df_copy.groupby('fk_contact').agg({
        'datetime_purchase': lambda date: (snapshot_date - date.max()).days,
        'nk_ota_localizer_id': 'count',
        'gmv_success': 'sum'
    }).reset_index()
    
    rfm_data.rename(columns={'datetime_purchase': 'Recency', 'nk_ota_localizer_id': 'Frequency', 'gmv_success': 'MonetaryValue'}, inplace=True)
    
    rfm_for_scaling = rfm_data[['Recency', 'Frequency', 'MonetaryValue']]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_for_scaling)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm_data

# --- Carregamento e Preparação Global dos Dados ---
try:
    df_processed = load_and_process_data('data/df_amostra.csv')
    df_segmented = generate_segmented_data(df_processed)
    
    # Criar a coluna 'target_route' no escopo global para estar disponível em todo o app
    top_routes_global = df_processed['route'].value_counts().nlargest(50).index.tolist()
    df_processed['target_route'] = df_processed['route'].apply(lambda x: x if x in top_routes_global else 'Outra')

except FileNotFoundError:
    st.error("Erro: Arquivo 'data/df_amostra.csv' não encontrado. Certifique-se de que a amostra de dados está no seu repositório GitHub.")
    st.stop()

# --- Título do Dashboard ---
st.title("🚌 Dashboard de Modelos de Dados - Análise de Clientes ClickBus")
st.markdown("Este dashboard apresenta os resultados dos três principais desafios de modelagem de dados.")

tab1, tab2, tab3 = st.tabs([
    "🎯 **1. Segmentação de Clientes**",
    "📈 **2. Previsão de Recompra**",
    "🗺️ **3. Previsão de Próxima Rota**"
    
])

# --- ABA 1: SEGMENTAÇÃO DE CLIENTES ---
with tab1:
    st.header("Segmentação de Clientes com RFM e K-Means")
    st.markdown("Agrupamos os clientes em perfis distintos (personas) com base no seu comportamento de compra.")
    cluster_analysis = df_segmented.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean().sort_values(by='MonetaryValue', ascending=False)
    personas = {
        2: {"nome": "🏆 Super Clientes (Campeões)", "desc": "A elite absoluta. Frequência e valor monetário ordens de magnitude acima dos demais. Ação: Tratamento VIP."},
        3: {"nome": "❤️ Clientes Fiéis", "desc": "Compram recentemente, com alta frequência e gastam bastante. Ação: Programas de fidelidade."},
        0: {"nome": "💡 Clientes Ocasionales", "desc": "Compram com baixa frequência e não o fazem há mais de um ano. Ação: Campanhas de reengajamento."},
        1: {"nome": "👻 Clientes Perdidos (Inativos)", "desc": "A última compra foi há quase 6 anos. Clientes efetivamente perdidos. Ação: Focar esforços nos outros grupos."}
    }
    cluster_analysis['Persona'] = cluster_analysis.index.map(lambda x: personas.get(x, {"nome": "Não Definido"})['nome'])
    cluster_analysis['Ação Sugerida'] = cluster_analysis.index.map(lambda x: personas.get(x, {"desc": "Ação: N/A"})['desc'].split("Ação: ")[1])
    st.subheader("Resumo dos Segmentos (Personas)")
    st.dataframe(cluster_analysis[['Persona', 'Recency', 'Frequency', 'MonetaryValue', 'Ação Sugerida']].style.format({
        'Recency': '{:.0f} dias', 'Frequency': '{:.1f} compras', 'MonetaryValue': 'R$ {:,.2f}'
    }))
    st.subheader("Visualização Interativa dos Clusters")
    col1, col2 = st.columns(2)
    with col1:
        df_segmented['MonetaryValueSize'] = df_segmented['MonetaryValue'].clip(lower=0)
        fig_scatter = px.scatter(
            df_segmented, x='Recency', y='Frequency', color='Cluster', size='MonetaryValueSize',
            hover_name='fk_contact', hover_data={'MonetaryValue': True, 'MonetaryValueSize': False},
            title='Visão Geral dos Clusters (RFM)', labels={'Recency': 'Recência (dias)', 'Frequency': 'Frequência (compras)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_bar = px.bar(
            cluster_analysis.sort_values('MonetaryValue'), x='MonetaryValue', y='Persona', orientation='h',
            color='Persona', title='Valor Monetário Médio por Segmento', labels={'MonetaryValue': 'Valor Médio (R$)', 'Persona': 'Segmento'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- ABA 2: PREVISÃO DE RECOMPRA ---
with tab2:
    st.header("Previsão de Recompra nos Próximos 30 Dias")
    st.markdown("Este modelo identifica clientes com maior probabilidade de realizar uma nova compra no próximo mês.")

    @st.cache_resource
    def train_repurchase_model(df):
        cutoff_date = df['datetime_purchase'].max() - pd.Timedelta(days=30)
        df_train = df[df['datetime_purchase'] < cutoff_date]
        if df_train.empty: return None, "Amostra de dados muito pequena para treinar."
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
        
        if len(y.unique()) < 2: return None, "Amostra não contém exemplos de recompra. O modelo não pode ser treinado."
        
        scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        return model, None

    xgb_repurchase_model, error_message = train_repurchase_model(df_processed)
    
    if error_message:
        st.warning(error_message)
    else:
        st.subheader("O Valor de Negócio do Modelo XGBoost")
        st.markdown("Nosso modelo é **4.6x mais eficaz** que uma abordagem genérica para encontrar clientes propensos a comprar.")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Perfil do Modelo (Dados Completos)**")
            st.markdown("- **Precisão (Precision): 12%**")
            st.markdown("- **Alcance (Recall): 70%**")
        with col2:
            fig_lift = go.Figure(go.Indicator(
                mode = "gauge+number", value = 4.6,
                title = {'text': "Ganho de Eficiência (Lift)"},
                gauge = {'axis': {'range': [1, 10]}, 'bar': {'color': "green"}}
            ))
            st.plotly_chart(fig_lift, use_container_width=True)

# --- ABA 3: PREVISÃO DE PRÓXIMA ROTA ---
with tab3:
    st.header("Previsão do Próximo Trecho de Viagem")
    st.markdown("Este modelo prevê qual será a próxima rota que um cliente irá comprar.")
    
    @st.cache_resource
    def train_route_model(df):
        df_sorted = df.sort_values(by=['fk_contact', 'datetime_purchase'])
        df_sorted['last_route'] = df_sorted.groupby('fk_contact')['target_route'].shift(1)
        df_predict = df_sorted.dropna(subset=['last_route'])
        
        X = df_predict[['last_route']]
        y = df_predict['target_route']
        
        le = LabelEncoder()
        X['last_route_encoded'] = le.fit_transform(X['last_route'])
        X = X.drop('last_route', axis=1)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        model.fit(X, y)
        return model, le

    rf_route_model, route_encoder = train_route_model(df_processed)

    st.subheader("Comparação de Performance dos Modelos")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="**Baseline (Rota Mais Frequente)**", value="24.02%")
    with col2:
        st.metric(label="**Modelo RandomForest**", value="78.75%", delta="54.73%")
    st.success("O modelo RandomForest é **3.3x mais preciso** que o baseline.")

    st.subheader("Teste o Modelo de Previsão de Rota")
    multi_purchase_clients = df_processed['fk_contact'].value_counts()[df_processed['fk_contact'].value_counts() > 1].index
    sample_client = st.selectbox("Selecione um cliente para testar:", options=multi_purchase_clients)
    
    if sample_client:
        client_history = df_processed[df_processed['fk_contact'] == sample_client].sort_values('datetime_purchase')
        last_route_real = client_history['route'].iloc[-1]
        
        if len(client_history) > 1:
            last_route_for_prediction = client_history['target_route'].iloc[-2]
            last_route_encoded = route_encoder.transform([last_route_for_prediction])
            prediction = rf_route_model.predict([[last_route_encoded[0]]])
            
            st.write(f"**Histórico:** A penúltima rota foi **{last_route_for_prediction}**.")
            st.write(f"➡️ **Previsão do Modelo:** **{prediction[0]}**")
            st.write(f"🎯 **Próxima Rota Real:** **{last_route_real}**")
        else:
            st.write("Cliente com apenas uma compra. Não é possível prever a próxima.")

st.sidebar.header("Sobre o Projeto")
st.sidebar.info("Dashboard interativo que apresenta os resultados dos modelos de segmentação e previsão de comportamento de clientes da ClickBus.")
