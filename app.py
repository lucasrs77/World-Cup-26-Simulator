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
    
    # 1. Preprocesar los datos
    df_probs_reset = df_probs.reset_index().rename(columns={'index': 'Team'})
    df_probs_reset.insert(1, 'Start', 100.0)
    stages = ['Start', 'Make R32', 'Make R16', 'Make QF', 'Make SF', 'Make Final', 'Win World Cup']
    
    df_melted = df_probs_reset.melt(
        id_vars=['Team'], 
        value_vars=stages,
        var_name='Stage', 
        value_name='Probability'
    )
    
    # 2. Diccionario de Grupos para el Filtro
    # Extraemos a qué grupo pertenece cada equipo leyendo los partidos
    team_groups = pd.concat([
        df_groups[['Team_A', 'Group']].rename(columns={'Team_A': 'Team'}),
        df_groups[['Team_B', 'Group']].rename(columns={'Team_B': 'Team'})
    ]).drop_duplicates().set_index('Team')['Group'].to_dict()
    
    # 3. Filtros Interactivos (Grupo y Equipos)
    col1, col2 = st.columns(2)
    
    with col1:
        all_groups = sorted(list(set(team_groups.values())))
        selected_groups = st.multiselect("Filter by Group (Optional):", options=all_groups)
    
    # Si seleccionó grupos, filtramos la lista de equipos disponibles. Si no, mostramos todos.
    if selected_groups:
        available_teams = sorted([t for t, g in team_groups.items() if g in selected_groups])
    else:
        available_teams = sorted(list(team_groups.keys()))
        
    with col2:
        selected_teams = st.multiselect(
            "Select Teams to compare:",
            options=available_teams,
            default=available_teams[:4] if available_teams else []
        ) # Límite de 8 removido
    
    if not selected_teams:
        st.warning("Please select at least one team.")
    else:
        df_filtered = df_melted[df_melted['Team'].isin(selected_teams)]
        
        # 4. Gráfico Plotly
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
        
        fig_survival.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_survival.update_layout(
            yaxis=dict(range=[0, 105], title_font=dict(size=14)),
            xaxis=dict(title="", tickfont=dict(size=12)),
            legend_title_text='National Team',
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_survival, use_container_width=True)
        
        with st.expander("Show raw probability matrix"):
            st.dataframe(
                df_probs.loc[selected_teams, stages[1:]].style.format("{:.1f}%")
               .background_gradient(cmap='Blues', axis=1),
               use_container_width=True
            )


# ==============================================================================
# TAB 2: GROUP VOLATILITY (THE 'GROUP OF DEATH' FINDER)
# ==============================================================================
with tab2:
    st.subheader("Group Volatility Index (The 'Group of Death' Finder)")
    st.write("By calculating the average statistical certainty of match outcomes within each group, we can identify which groups are highly predictable and which are absolute bloodbaths.")
    
    # 1. Feature Engineering
    df_groups['Max_Prob'] = df_groups[['Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%']].max(axis=1)
    df_volatility = df_groups.groupby('Group')['Max_Prob'].mean().reset_index()
    df_volatility.rename(columns={'Max_Prob': 'Predictability_Score'}, inplace=True)
    df_volatility['Volatility_Index'] = 100 - df_volatility['Predictability_Score']
    df_volatility = df_volatility.sort_values('Volatility_Index', ascending=False).reset_index(drop=True)
    
    # 2. Gráfico
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

    '''
    # 3. Inspector de Grupos Corregido (Sin Score y con Decimales formateados)
    st.markdown("### 🔍 Inspect Group Matches")
    st.markdown("*Note: This table strips away the noise and displays pure win/draw/loss probability distributions. Groups saturated with deep red across multiple matches indicate a high-variance 'Group of Death' scenario.*")
    
    selected_group = st.selectbox("Select a Group to see its internal match probabilities:", df_volatility['Group'].unique())
    
    df_group_matches = df_groups[df_groups['Group'] == selected_group][
        ['Team_A', 'Team_B', 'Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%']
    ]
    
    # Formateo corporativo: 1 decimal, agrega el %, y pinta la matriz
    st.dataframe(
        df_group_matches.style
        .format({
            'Prob_Win_A_%': "{:.1f}%", 
            'Prob_Draw_%': "{:.1f}%", 
            'Prob_Win_B_%': "{:.1f}%"
        })
        .background_gradient(cmap='Reds', subset=['Prob_Win_A_%', 'Prob_Draw_%', 'Prob_Win_B_%']),
        use_container_width=True,
        hide_index=True
    )
    '''

# ==============================================================================
# TAB 3: THE ALGORITHM LAB (DETERMINISTIC ENGINE)
# ==============================================================================
with tab3:
    st.subheader("The Algorithm Lab: Deterministic Power Rankings")
    st.write("Football philosophy debate: What wins World Cups, an elite attack or an impenetrable defense? Adjust the strategic weights below to recalculate the deterministic 'Power Score' for each nation in real-time.")

    # 1. Controles de Laboratorio (Sliders)
    st.markdown("### ⚙️ Strategic Weighting")
    
    # El usuario elige cuánto pesa el ataque. La defensa es el resto (1 - ataque)
    off_weight = st.slider("Offensive Importance Weight (%)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    def_weight = 1.0 - off_weight
    
    st.markdown(f"**Current Algorithm Philosophy:** {off_weight*100:.0f}% Attack Focus / {def_weight*100:.0f}% Defense Focus")
    st.divider()

    # 2. Motor Determinístico en Vivo
    # Formula: Power = (Attack * W_off) + (Defense * W_def)
    # Como tu Defensa ajustada es mejor cuanto más BAJA es (menos goles recibe), 
    # tenemos que invertirla para el Power Score (1 / Defense)
    
    df_lab = df_strengths.copy()
    
    # Invertimos la defensa de forma segura para que "número más alto = mejor"
    df_lab['Inverted_Defense'] = 1.0 / df_lab['Adjusted_Defense'].replace(0, 0.01)
    
    # Calculamos el Power Score
    df_lab['Power_Score'] = (df_lab['Adjusted_Attack'] * off_weight) + (df_lab['Inverted_Defense'] * def_weight)
    
    # Limpiamos y ordenamos el Top 10
    df_lab = df_lab.sort_values('Power_Score', ascending=False).reset_index(drop=True)
    df_lab.index += 1 # Para que el ranking empiece en 1 y no en 0
    
    # 3. Visualización Corporativa
    st.markdown("### 🏆 Top 10 Deterministic Favorites")
    st.markdown("*Based on your selected algorithmic weights.*")
    
    st.dataframe(
        df_lab[['Team', 'Adjusted_Attack', 'Adjusted_Defense', 'Power_Score']].head(10)
        .style
        .format({
            'Adjusted_Attack': "{:.2f}", 
            'Adjusted_Defense': "{:.2f}", 
            'Power_Score': "{:.3f}"
        })
        .background_gradient(cmap='Greens', subset=['Power_Score']),
        use_container_width=True
    )



    
