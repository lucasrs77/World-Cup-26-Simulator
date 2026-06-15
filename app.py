import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import poisson

# ==============================================================================
# 0. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="World Cup 2026: Strategic Data Lab",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { 
        height: 48px; 
        font-weight: bold; 
        border-radius: 4px 4px 0px 0px;
    }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Survival Analytics", 
    "🔥 Group Volatility", 
    "🧪 The Algorithm Lab", 
    "📊 Match Profiler",
    "🔍 Exploratory Data Analysis (EDA)"
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
        # Checkbox estratégico para hacer zoom en la cola de la distribución
        use_log_scale = st.checkbox("🔍 Use Logarithmic Scale (Zoom in on Final/Win probabilities)")
        
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
        
        # Aplicamos suavizado 'spline' para curvar las líneas elegantemente
        fig_survival.update_traces(line=dict(width=3, shape='spline'), marker=dict(size=8))
        
        # Ajuste dinámico del Eje Y
        if use_log_scale:
            # Escala logarítmica para separar los valores pequeños
            fig_survival.update_layout(yaxis_type="log", yaxis_title="Likelihood (%) - Log Scale")
        else:
            # Rango normal, bajando el piso a -2 para que la línea del 0% no quede cortada por el borde
            fig_survival.update_layout(yaxis=dict(range=[-2, 105], title_font=dict(size=14)))
            
        fig_survival.update_layout(
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

# ==============================================================================
# TAB 3: THE ALGORITHM LAB (DETERMINISTIC ENGINE)
# ==============================================================================
with tab3:
    st.subheader("The Algorithm Lab: Custom Strategic Projections")
    st.write("Play God with the model parameters. Adjust the weights of historical performance versus financial power, modify tactical focus, and apply environmental boosts to recalculate the global tournament hierarchy in real-time.")

    # 1. CONTROLES DEL LABORATORIO (SLIDERS)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📊 Core Weights")
        # Slider principal: Historia vs Billetera
        history_weight = st.slider(
            "Historical Data Weight (%)", 
            min_value=0.0, max_value=1.0, value=0.60, step=0.05,
            help="At 100%, squad market value is ignored. At 0%, only team finances matter."
        )
        financial_weight = 1.0 - history_weight
        
        st.markdown("### ⚙️ Tactical Philosophy")
        off_weight = st.slider(
            "Offensive Importance Weight (%)", 
            min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="Balances the importance of scoring capability versus defensive solidity in the final score."
        )
        def_weight = 1.0 - off_weight

    with col_b:
        st.markdown("### 🌎 Environmental Boosts")
        americas_boost = st.slider(
            "Americas Home Advantage Boost (%)", 
            min_value=0.0, max_value=20.0, value=5.0, step=1.0,
            help="Stat multiplier applied to CONMEBOL and CONCACAF teams due to climate, travel, and local support."
        )
        
        st.markdown("### 🎲 The Chaos Factor")
        chaos_factor = st.slider(
            "Underdog Competitiveness Multiplier", 
            min_value=1.0, max_value=2.0, value=1.0, step=0.1,
            help="Values > 1.0 mathematically compress the skill gap, buffing weaker teams and simulating high volatility."
        )
        
        # Lista de referencia para el boost regional
        americas_teams = ['Argentina', 'Brazil', 'Uruguay', 'Colombia', 'Ecuador', 
                          'USA', 'Mexico', 'Canada', 'Peru', 'Chile', 'Venezuela', 
                          'Paraguay', 'Costa Rica', 'Panama', 'Jamaica']
    
    st.divider()

    # 2. MOTOR DE SIMULACIÓN DETERMINÍSTICO EN VIVO
    df_lab = df_strengths.copy()
    
    # Normalizamos el Market Value a una escala similar a las strengths (0.5 a 2.5) para poder mezclarlos
    # Aplicamos raíz cuadrada (the Nate Silver approach) para evitar que las brechas de dinero destruyan la lógica
    df_lab['Financial_Score'] = np.sqrt(df_lab['Market_Value'] / df_lab['Market_Value'].mean())
    
    # Calculamos el Ataque y Defensa dinámico mezclando Historia y Billetera
    df_lab['Dynamic_Attack'] = (df_lab['Historical_Attack'] * history_weight) + (df_lab['Financial_Score'] * financial_weight)
    
    # Para la defensa, como número más bajo es mejor, el score financiero resta vulnerabilidad
    df_lab['Dynamic_Defense'] = (df_lab['Historical_Defense'] * history_weight) + ((2.0 - df_lab['Financial_Score']) * financial_weight)
    df_lab['Dynamic_Defense'] = df_lab['Dynamic_Defense'].clip(lower=0.1) # Evitar que baje de cero
    
    # Invertimos la defensa para el Power Score final (menor defensa = mayor poder)
    df_lab['Inverted_Defense'] = 1.0 / df_lab['Dynamic_Defense']
    
    # Combinamos según la filosofía táctica (Ataque vs Defensa)
    df_lab['Base_Power'] = (df_lab['Dynamic_Attack'] * off_weight) + (df_lab['Inverted_Defense'] * def_weight)
    
    # Aplicamos el Boost de las Américas
    boost_multiplier = 1.0 + (americas_boost / 100.0)
    df_lab['Power_Score'] = np.where(
        df_lab['Team'].isin(americas_teams), 
        df_lab['Base_Power'] * boost_multiplier, 
        df_lab['Base_Power']
    )
    
    # Aplicamos el Factor Caos (Comprimir la brecha de talento)
    df_lab['Power_Score'] = df_lab['Power_Score'] ** (1.0 / chaos_factor)
    
    # Ordenamos el ranking completo
    df_lab = df_lab.sort_values('Power_Score', ascending=False).reset_index(drop=True)
    df_lab.index += 1 # El ranking arranca en 1
    
    # 3. CONTROL DE VISUALIZACIÓN DINÁMICO (SELECTOR DE CANTIDAD DE EQUIPOS)
    st.markdown("### 🏆 Adjusted Power Rankings")
    
    max_teams_to_show = st.slider(
        "Select number of teams to display in the ranking:", 
        min_value=5, 
        max_value=48, 
        value=15, 
        step=1
    )
    
    # Renderizado de la tabla BI corporativa
    st.dataframe(
        df_lab[['Team', 'Dynamic_Attack', 'Dynamic_Defense', 'Power_Score']].head(max_teams_to_show)
        .style
        .format({
            'Dynamic_Attack': "{:.2f}", 
            'Dynamic_Defense': "{:.2f}", 
            'Power_Score': "{:.3f}"
        })
        .background_gradient(cmap='Greens', subset=['Power_Score']),
        use_container_width=True
    )


# ==============================================================================
# TAB 4: MATCH PROFILER (POISSON HEATMAP)
# ==============================================================================
with tab4:
    st.subheader("Match Profiler: Bivariate Poisson Distribution")
    st.write("A purely statistical view of match dynamics. By calculating the expected goals (xG) based on historical strength and applying a Dixon-Coles adjustment for low-scoring matches, we map the exact probability of every possible scoreline.")

    # 1. Selectores de Equipos
    teams_list = sorted(df_strengths['Team'].unique())
    col1, col2 = st.columns(2)
    
    with col1:
        team_a = st.selectbox("Select Team A (Home):", teams_list, index=teams_list.index('Argentina') if 'Argentina' in teams_list else 0)
    with col2:
        # Filtramos para que no pueda elegir el mismo equipo de ambos lados
        available_team_b = [t for t in teams_list if t != team_a]
        team_b = st.selectbox("Select Team B (Away):", available_team_b, index=available_team_b.index('France') if 'France' in available_team_b else 0)

    st.divider()

    # 2. Extracción de Fuerzas y Cálculo de xG (Lambdas)
    stats_a = df_strengths[df_strengths['Team'] == team_a].iloc[0]
    stats_b = df_strengths[df_strengths['Team'] == team_b].iloc[0]

    GLOBAL_AVG_GOALS = 1.45 # Constante global de goles del Mundial

    # Fórmula: Promedio Global * Ataque Propio * Defensa Rival
    lambda_a = GLOBAL_AVG_GOALS * stats_a['Historical_Attack'] * stats_b['Historical_Defense']
    lambda_b = GLOBAL_AVG_GOALS * stats_b['Historical_Attack'] * stats_a['Historical_Defense']

    # 3. Construcción de la Matriz de Poisson (0 a 5 goles)
    max_goals = 5
    goals = np.arange(max_goals + 1)
    
    prob_a = poisson.pmf(goals, lambda_a)
    prob_b = poisson.pmf(goals, lambda_b)
    prob_matrix = np.outer(prob_a, prob_b)

    # Ajuste Dixon-Coles (rho = 0.05) para deflactar 0-0 y 1-1
    rho = 0.05
    prob_matrix[0, 0] *= (1 - lambda_a * lambda_b * rho)
    prob_matrix[0, 1] *= (1 + lambda_a * rho)
    prob_matrix[1, 0] *= (1 + lambda_b * rho)
    prob_matrix[1, 1] *= (1 - rho)
    
    prob_matrix = np.maximum(0, prob_matrix) # Evitar probabilidades negativas
    prob_matrix /= prob_matrix.sum() # Renormalizar a 100%

    # 4. Visualización: El Mapa de Calor (Heatmap)
    col_heat, col_stats = st.columns([2, 1])

    with col_heat:
        st.markdown(f"### 🌡️ Scoreline Probability Matrix")
        
        # Formatear la matriz para que muestre porcentajes legibles
        text_matrix = [[f"{val*100:.1f}%" for val in row] for row in prob_matrix]

        fig_heat = go.Figure(data=go.Heatmap(
            z=prob_matrix,
            x=[f"{team_b} {i}" for i in range(max_goals + 1)],
            y=[f"{team_a} {i}" for i in range(max_goals + 1)],
            text=text_matrix,
            texttemplate="%{text}",
            colorscale='Reds',
            hoverinfo="text",
            showscale=False
        ))
        
        fig_heat.update_layout(
            xaxis_title=f"Goals {team_b}",
            yaxis_title=f"Goals {team_a}",
            yaxis_autorange='reversed', # Para que el 0-0 quede arriba a la izquierda
            template="plotly_dark", # Cambiado a Dark (o white si no usaste el CSS oscuro)
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)

    # 5. KPIs y Extracción de los Top Resultados
    with col_stats:
        st.markdown("### 🎯 Expected Goals (xG)")
        st.metric(label=f"{team_a} xG", value=f"{lambda_a:.2f}")
        st.metric(label=f"{team_b} xG", value=f"{lambda_b:.2f}")
        
        st.divider()
        
        # Aplanar la matriz para buscar los extremos
        flat_probs = prob_matrix.flatten()
        sorted_indices = np.argsort(flat_probs)[::-1]
        
        st.markdown("### 🔝 Most Likely Scores")
        for i in range(5):
            idx = sorted_indices[i]
            g_a, g_b = np.unravel_index(idx, prob_matrix.shape)
            prob_val = prob_matrix[g_a, g_b] * 100
            st.markdown(f"**{g_a} - {g_b}** ➔ {prob_val:.1f}%")
            
        st.divider()
        
        st.markdown("### 🦢 Statistical Black Swans")
        st.caption("Highly unlikely, yet mathematically possible (< 0.5%)")
        black_swans_count = 0
        
        # Recorremos desde el menos probable hacia arriba
        for i in range(len(flat_probs)-1, -1, -1):
            idx = sorted_indices[i]
            g_a, g_b = np.unravel_index(idx, prob_matrix.shape)
            prob_val = prob_matrix[g_a, g_b] * 100
            if 0.0 < prob_val < 0.5 and black_swans_count < 5:
                st.markdown(f"**{g_a} - {g_b}** ➔ {prob_val:.2f}%")
                black_swans_count += 1



# ==============================================================================
# TAB 5: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab5:
    st.subheader("Exploratory Data Analysis & Market Landscape")
    st.write("Understand the baseline features driving the simulation engine: historical team strengths, squad market values, and how the model stratifies contenders into tier hierarchies.")

    # --------------------------------------------------------------------------
    # VISUALIZACIÓN 1: CUADRANTE MÁGICO TÁCTICO (SCATTER PLOT)
    # --------------------------------------------------------------------------
    st.markdown("### 🗺️ Tactical Strength Quadrants")
    st.write("This space maps offensive capability against defensive solidity. The size of each bubble represents the squad's Market Value in millions.")
    
    df_scatter = df_strengths.copy()
    
    # Calculamos las medias para trazar las líneas de los cuadrantes
    mean_att = df_scatter['Historical_Attack'].mean()
    mean_def = df_scatter['Historical_Defense'].mean()
    
    # Construcción del Scatter Plot con Plotly
    fig_scatter = px.scatter(
        df_scatter,
        x="Historical_Attack",
        y="Historical_Defense",
        size="Market_Value",
        hover_name="Team",
        color="Historical_Attack", 
        color_continuous_scale="RdYlGn_r", 
        title="Team Strength Profile vs Market Value",
        labels={
            "Historical_Attack": "Historical Attack Strength (Higher = Better)",
            "Historical_Defense": "Historical Defense Strength (Lower = Better)"
        },
        template="plotly_dark" # Cambiado a Dark (o white si no usaste el CSS oscuro)
    )
    
    # Invertir el eje Y de defensa porque "número más bajo es mejor"
    fig_scatter.update_yaxes(autorange="reversed")    
    
    # Añadir las líneas punteadas que dividen los 4 cuadrantes tácticos
    fig_scatter.add_vline(x=mean_att, line_width=1.5, line_dash="dash", line_color="gray")
    fig_scatter.add_hline(y=mean_def, line_width=1.5, line_dash="dash", line_color="gray")
    
    # Quitar la barra de color lateral
    fig_scatter.update_layout(coloraxis_showscale=False)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # EL ARREGLO CRÍTICO: Uso de """ para strings multilínea en Streamlit
    col_ed1, col_ed2, col_ed3, col_ed4 = st.columns(4)
    col_ed1.info("""**Upper Right: Elite**
    High scoring power combined with a highly compact defense.""")
    col_ed2.warning("""**Lower Right: Attacking Focus**
    High scoring metrics but structurally vulnerable at the back.""")
    col_ed3.success("""**Upper Left: Defensive Wall**
    Impenetrable defensive setup but lacks clinical finishing.""")
    col_ed4.error("""**Lower Left: Underdogs**
    Sub-average baseline metrics across both phases.""")
    
    st.divider()

    # --------------------------------------------------------------------------
    # VISUALIZACIÓN 2: TIER LIST COMPETITIVA (TREEMAP)
    # --------------------------------------------------------------------------
    st.markdown("### 👑 Tournament Tier Hierarchy")
    st.write("Instead of looking at raw singular percentages, this Treemap segments the 48 nations into distinct competitive tiers based on their objective probability of winning the World Cup.")
    
    # Procesamos df_probs para armar los Tiers
    df_tiers = df_probs[['Win World Cup']].reset_index().rename(columns={'index': 'Team'})
    
    # 1. Encontrar el techo dinámico
    max_prob = df_tiers['Win World Cup'].max()
    
    # 2. Definir los cortes dinámicos
    t1_threshold = max_prob * 0.60  
    t2_threshold = max_prob * 0.33  
    t3_threshold = 0.50             
    
    # 3. Función de categorización
    def assign_tier_category_dynamic(prob):
        if prob >= t1_threshold:
            return f"Tier 1: Title Contenders (>= {t1_threshold:.1f}%)"
        elif prob >= t2_threshold:
            return f"Tier 2: Strong Contenders (>= {t2_threshold:.1f}%)"
        elif prob >= t3_threshold:
            return f"Tier 3: Knockout Hopefuls (>= {t3_threshold:.1f}%)"
        else:
            return f"Tier 4: The Underdogs (< {t3_threshold:.1f}%)"
            
    df_tiers['Tier'] = df_tiers['Win World Cup'].apply(assign_tier_category_dynamic)
    
    # Construcción del Treemap interactivo
    fig_tree = px.treemap(
        df_tiers,
        path=['Tier', 'Team'], 
        values='Win World Cup',
        color='Win World Cup',
        color_continuous_scale='Blues',
        title="Contenders Stratification Matrix",
        labels={'Win World Cup': 'Chances of Winning (%)'}
    )
    
    fig_tree.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        template="plotly_dark" # Cambiado a Dark (o white si no usaste el CSS oscuro)
    )
    
    st.plotly_chart(fig_tree, use_container_width=True)
