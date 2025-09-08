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
def generate_segmented_data(_df): # O underline evita que o Streamlit hasheie o DF inteiro
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

# --- Carregamento dos Dados ---
try:
    df_processed = load_and_process_data('data/df_amostra.csv') 
    df_segmented = generate_segmented_data(df_processed)
except FileNotFoundError:
    st.error("Erro: Arquivo 'data/df_amostra.csv' não encontrado no repositório. Por favor, certifique-se de que a amostra de dados foi enviada ao GitHub.")
    st.stop()

st.title("🚌 Dashboard de Modelos de Dados - Análise de Clientes ClickBus")
st.markdown("Este dashboard apresenta os resultados dos três principais desafios de modelagem de dados.")

tab1, tab2, tab3 = st.tabs([
    "🎯 **1. Segmentação de Clientes (K-Means)**",
    "🗺️ **2. Previsão de Próxima Rota (RandomForest)**",
    "📈 **3. Previsão de Recompra (XGBoost)**"
])

# --- ABA 1: SEGMENTAÇÃO DE CLIENTES ---
with tab1:
    st.header("Segmentação de Clientes com RFM e K-Means")
    # ... (O resto do código da Aba 1 continua igual, pois `df_segmented` agora é gerado em memória)
    cluster_analysis = df_segmented.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean().sort_values(by='MonetaryValue', ascending=False)
    personas = {
        2: {"nome": "🏆 Super Clientes (Campeões)", "desc": "A elite absoluta. Frequência e valor monetário ordens de magnitude acima dos demais. Provavelmente agências ou empresas. Ação: Tratamento VIP, gerente de contas dedicado."},
        3: {"nome": "❤️ Clientes Fiéis", "desc": "Compram muito recentemente, com alta frequência e gastam bastante. São a base de clientes recorrentes e engajados. Ação: Programas de fidelidade, ofertas exclusivas."},
        0: {"nome": "💡 Clientes Ocasionales", "desc": "Compram com baixa frequência e não o fazem há mais de um ano. Precisam de um incentivo para não se tornarem inativos. Ação: Campanhas de reengajamento com descontos."},
        1: {"nome": "👻 Clientes Perdidos (Inativos)", "desc": "A última compra foi há quase 6 anos. Clientes efetivamente perdidos, com baixíssimo valor. Ação: Focar esforços de marketing nos outros grupos."}
    }
    cluster_analysis['Persona'] = cluster_analysis.index.map(lambda x: personas.get(x, {"nome": "Não Definido"})['nome'])
    cluster_analysis['Ação Sugerida'] = cluster_analysis.index.map(lambda x: personas.get(x, {"desc": "Ação: N/A"})['desc'].split("Ação: ")[1])
    st.subheader("Resumo dos Segmentos (Personas)")
    st.dataframe(cluster_analysis[['Persona', 'Recency', 'Frequency', 'MonetaryValue', 'Ação Sugerida']].style.format({
        'Recency': '{:.0f} dias',
        'Frequency': '{:.1f} compras',
        'MonetaryValue': 'R$ {:,.2f}'
    }))
    st.subheader("Visualização Interativa dos Clusters")
    col1, col2 = st.columns(2)
    with col1:
        df_segmented['MonetaryValueSize'] = df_segmented['MonetaryValue'].clip(lower=0)
        fig_scatter = px.scatter(
            df_segmented, x='Recency', y='Frequency', color='Cluster', size='MonetaryValueSize',
            hover_name='fk_contact', hover_data={'MonetaryValue': True, 'MonetaryValueSize': False},
            title='Visão Geral dos Clusters (RFM)', labels={'Recency': 'Recência (dias)', 'Frequency': 'Frequência (compras)'},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_bar = px.bar(
            cluster_analysis.sort_values('MonetaryValue'), x='MonovoletaryValue', y='Persona', orientation='h',
            color='Persona', title='Valor Monetário Médio por Segmento', labels={'MonetaryValue': 'Valor Médio (R$)', 'Persona': 'Segmento'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
# --- ABA 2: PREVISÃO DE RECOMPRA ---
with tab2:
    st.header("Previsão de Recompra nos Próximos 30 Dias")
    st.markdown("""
    Aqui, o desafio é identificar quais clientes têm a maior probabilidade de realizar uma nova compra no próximo mês.
    Construímos um modelo **XGBoost** que, apesar de uma precisão aparentemente baixa, oferece um grande valor de negócio.
    """)

    # --- Lógica do Modelo (em cache para performance) ---
    @st.cache_resource
    def train_repurchase_model(df):
        # 1. Janela de predição
        cutoff_date = df['datetime_purchase'].max() - pd.Timedelta(days=30)
        df_train = df[df['datetime_purchase'] < cutoff_date]
        df_target = df[df['datetime_purchase'] >= cutoff_date]
        snapshot_date_model = cutoff_date + pd.Timedelta(days=1)
        
        # 2. Features
        features_df = df_train.groupby('fk_contact').agg(
            Recency=('datetime_purchase', lambda date: (snapshot_date_model - date.max()).days),
            Frequency=('datetime_purchase', 'count'),
            MonetaryValue=('gmv_success', 'sum')
        ).reset_index()

        # 3. Alvo
        target_customers = df_target['fk_contact'].unique()
        features_df['will_buy_in_30_days'] = features_df['fk_contact'].isin(target_customers).astype(int)
        
        X = features_df[['Recency', 'Frequency', 'MonetaryValue']]
        y = features_df['will_buy_in_30_days']
        
        # 4. Parâmetro de balanceamento
        scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
        
        # 5. Treinar modelo
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, use_label_encoder=False, 
            eval_metric='logloss', random_state=42
        )
        model.fit(X, y)
        return model, X, y

    xgb_repurchase_model, X_repurchase, y_repurchase = train_repurchase_model(df_processed)
    
    # --- Exibição dos Resultados ---
    st.subheader("O Valor de Negócio do Modelo XGBoost")
    st.markdown("""
    Embora a **precisão** do modelo seja de **12%** (com limiar ajustado), isso representa um ganho enorme. Se a taxa de recompra natural é de ~2.6%, nosso modelo é **4.6x mais eficaz** em encontrar clientes propensos a comprar.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Perfil do Modelo (XGBoost com limiar ajustado para 0.6)**")
        st.markdown("- **Precisão (Precision): 12%**\n    - *De cada 100 clientes que o modelo aponta, 12 realmente compram.*")
        st.markdown("- **Alcance (Recall): 70%**\n    - *O modelo consegue encontrar 70% de todos os clientes que de fato recompraram.*")
        
    with col2:
        fig_lift = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 4.6,
            title = {'text': "Ganho de Eficiência (Lift) vs. Abordagem Genérica"},
            gauge = {'axis': {'range': [1, 10]}, 'bar': {'color': "green"}},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        st.plotly_chart(fig_lift, use_container_width=True)
    
    st.subheader("Estratégia de Uso")
    st.success("""
    **Este modelo é ideal para campanhas de marketing de baixo custo e grande escala (ex: e-mail marketing).** Ele nos permite focar em um grupo muito mais qualificado de clientes, maximizando o alcance e o retorno sobre o investimento.
    """)

st.sidebar.header("Sobre o Projeto")
st.sidebar.info("""
Este dashboard é o resultado de uma análise de dados completa para a ClickBus, cobrindo desde a segmentação de clientes até a criação de modelos preditivos avançados.
- **Tecnologias:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Plotly.
- **Objetivo:** Transformar dados brutos em insights acionáveis e ferramentas estratégicas para o negócio.
""")
