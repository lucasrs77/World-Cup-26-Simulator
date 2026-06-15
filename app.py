import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="World Cup 2026: Strategic Data Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para sacar la vibra de "casino"
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ FIFA World Cup 2026: Strategic Data Lab")
st.markdown("""
**Methodology & Limitations:** This lab utilizes a Bivariate Poisson model adjusted with a Dixon-Coles parameter ($\\rho = 0.05$) to handle low-scoring draw overdispersion. Team strengths are a dynamic SPI blend (60% Historical Time-Decay + 40% Transfermarkt Squad Value). Tournament projections are derived from a 10,000-iteration Monte Carlo engine. *Note: Football possesses inherent stochastic variance; these are statistical likelihoods, not certainties.*
""")

# ==============================================================================
# 1. DATA INGESTION (CACHED)
# ==============================================================================
@st.cache_data
def load_data():
    # Usá las URLs crudas (raw) de tu repositorio de GitHub
    base_url = "https://raw.githubusercontent.com/lucasrs77/World-Cup-26-Simulator/main/"
    
    df_probs = pd.read_csv(base_url + "WC26_Probabilities.csv", index_col=0)
    df_groups = pd.read_csv(base_url + "WC26_GroupStage_Predictions.csv")
    df_strengths = pd.read_csv(base_url + "WC26_Team_Strengths.csv")
    
    return df_probs, df_groups, df_strengths

try:
    df_probs, df_groups, df_strengths = load_data()
except Exception as e:
    st.error(f"Error cargando los datos de GitHub. Verificá que los archivos existan. Detalle: {e}")
    st.stop()

# ==============================================================================
# 2. ESTRUCTURA DE PESTAÑAS (TABS)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Survival Analytics", 
    "🔥 Group Volatility", 
    "🧪 The Algorithm Lab", 
    "📊 Match Profiler"
])



# ==============================================================================
# TAB 1: SURVIVAL CURVE (ANALÍTICA DE SUPERVIVENCIA)
# ==============================================================================
with tab1:
    st.subheader("Tournament Survival Curve")
    st.write("Analyze the probability decay of teams advancing through the knockout stages.")
    
    # 1. Preprocesar los datos para Plotly (Formato Largo / Melt)
    # df_probs tiene los equipos en el índice. Lo pasamos a columna.
    df_probs_reset = df_probs.reset_index().rename(columns={'index': 'Team'})
    
    # Agregamos la fase inicial (100% de entrar al torneo) para que el gráfico empiece arriba
    df_probs_reset.insert(1, 'Start', 100.0)
    
    stages = ['Start', 'Make R32', 'Make R16', 'Make QF', 'Make SF', 'Make Final', 'Win World Cup']
    
    # Melt: Transforma las columnas de fases en filas
    df_melted = df_probs_reset.melt(
        id_vars=['Team'], 
        value_vars=stages,
        var_name='Stage', 
        value_name='Probability'
    )
    
    # 2. Selector de Equipos interactivo
    # Por defecto, ponemos algunos potentes para que el gráfico no esté vacío
    default_teams = ['Argentina', 'France', 'England', 'Morocco']
    all_teams = df_probs_reset['Team'].tolist()
    
    selected_teams = st.multiselect(
        "Select Teams to compare survival paths:",
        options=all_teams,
        default=default_teams,
        max_selections=8
    )
    
    if not selected_teams:
        st.warning("Please select at least one team.")
    else:
        # Filtrar datos
        df_filtered = df_melted[df_melted['Team'].isin(selected_teams)]
        
        # 3. Construir el gráfico Plotly
        fig_survival = px.line(
            df_filtered, 
            x='Stage', 
            y='Probability', 
            color='Team',
            markers=True,
            title="Probability of Reaching Each Stage (%)",
            labels={'Probability': 'Likelihood (%)', 'Stage': 'Tournament Stage'},
            template="plotly_white"
        )
        
        # Ajustes estéticos corporativos
        fig_survival.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_survival.update_layout(
            yaxis=dict(range=[0, 105], title_font=dict(size=14)),
            xaxis=dict(title="", tickfont=dict(size=12)),
            legend_title_text='National Team',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_survival, use_container_width=True)
        
        # 4. Tabla de datos crudos (opcional pero muy útil)
        with st.expander("Show raw probability matrix"):
            st.dataframe(
                df_probs.loc[selected_teams, stages[1:]].style.format("{:.1f}%")
               .background_gradient(cmap='Blues', axis=1)
            )




# ==============================================================================
# TAB 2: GROUP VOLATILITY (THE 'GROUP OF DEATH' FINDER)
# ==============================================================================
with tab2:
    st.subheader("Group Volatility Index (The 'Group of Death' Finder)")
    st.write("By calculating the average statistical certainty of match outcomes within each group, we can identify which groups are highly predictable and which are absolute bloodbaths.")
    
    # 1. Feature Engineering en vivo: Calcular la previsibilidad del grupo
    # La "certidumbre" de un partido es el valor máximo entre las prob de Local, Empate o Visitante
    df_groups['Max_Prob'] = df_groups[['Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%']].max(axis=1)
    
    # Promediamos la certidumbre por grupo
    df_volatility = df_groups.groupby('Group')['Max_Prob'].mean().reset_index()
    df_volatility.rename(columns={'Max_Prob': 'Predictability_Score'}, inplace=True)
    
    # 2. Invertir la escala para tener "Volatilidad" (100 - Previsibilidad)
    # A mayor número, más caos y menos favoritos claros.
    df_volatility['Volatility_Index'] = 100 - df_volatility['Predictability_Score']
    
    # Ordenar del más caótico al más predecible
    df_volatility = df_volatility.sort_values('Volatility_Index', ascending=False).reset_index(drop=True)
    
    # 3. Gráfico de Barras Horizontales con Plotly
    fig_vol = px.bar(
        df_volatility, 
        x='Volatility_Index', 
        y='Group', 
        orientation='h',
        title="Group Volatility Ranking",
        color='Volatility_Index',
        color_continuous_scale='Reds',
        text_auto='.1f'
    )
    
    fig_vol.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Volatility Index (Higher = More Chaos)",
        yaxis_title="Group",
        coloraxis_showscale=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # 4. Inspector de Grupos (Para ver por qué un grupo es tan volátil)
    st.markdown("### Inspect Group Matches")
    selected_group = st.selectbox("Select a Group to see its internal match probabilities:", df_volatility['Group'].unique())
    
    df_group_matches = df_groups[df_groups['Group'] == selected_group][
        ['Team_A', 'Team_B', 'Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%', 'Most_Likely_Score']
    ]
    
    st.dataframe(
        df_group_matches.style.background_gradient(cmap='Reds', subset=['Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%']),
        use_container_width=True,
        hide_index=True
    )
