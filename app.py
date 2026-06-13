import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="World Cup 2026 Simulator", page_icon="🏆", layout="wide")

st.title("🏆 FIFA World Cup 2026: AI & Monte Carlo Simulator")
st.markdown("""
Welcome to the ultimate predictive model for the 2026 World Cup. 
Powered by historical match data, dynamic FIFA rankings, Transfermarkt financial valuations, and a Bivariate Poisson distribution.
""")

# --- 2. CARGA DE DATOS (Caché para que sea rapidísimo) ---
# El decorador @st.cache_data hace que la app lea el CSV una sola vez y lo guarde en memoria
@st.cache_data
def load_probabilities():
    # Lee el archivo directamente desde tu mismo repositorio
    return pd.read_csv('WC26_Probabilities.csv', index_col=0)

df_probs = load_probabilities()

# --- 3. CREACIÓN DE LAS PESTAÑAS (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 The Oracle (10k Simulations)", "⚔️ Match Predictor", "🌌 Multiverse Explorer"])

with tab1:
    st.header("The Oracle: Who will lift the trophy?")
    st.markdown("We simulated the entire tournament **10,000 times**. Here are the mathematical probabilities:")
    
    # Mostramos el DataFrame interactivo
    st.dataframe(
        df_probs.style.format("{:.1f}%"), 
        use_container_width=True,
        height=600
    )

with tab2:
    st.header("Match Predictor")
    st.markdown("Select two teams to calculate their Expected Goals (xG) and most likely scoreline.")
    
    # Armamos dos columnas para que los selectores queden uno al lado del otro
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A:", ["Argentina", "Brazil", "France", "Spain", "England"])
    with col2:
        team_b = st.selectbox("Select Team B:", ["Morocco", "Japan", "Germany", "Portugal", "Senegal"])
    
    if st.button("Predict Match"):
        # ACÁ DESPUÉS VAMOS A METER TU FUNCIÓN DE POISSON
        st.success(f"Simulation complete! Analyzing {team_a} vs {team_b}...")
        st.info("The most logical score is: **1 - 0**") # Placeholder temporal

with tab3:
    st.header("Multiverse Explorer")
    st.markdown("Simulate **one** specific timeline of the 2026 World Cup.")
    
    user_seed = st.number_input("Enter a Seed (e.g., 1986 for Argentina's timeline):", value=1986, step=1)
    
    if st.button("Generate Timeline"):
        # ACÁ DESPUÉS VAMOS A METER TU MOTOR DE 103 PARTIDOS
        st.warning("Running 103 matches... (Coming soon)")
