# ⚽ FIFA World Cup 2026: Data Analysis & Monte Carlo Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://world-cup-26-simulator.streamlit.app/])

An end-to-end data science pipeline and interactive web application that simulates the expanded 48-team FIFA World Cup 2026. This project moves beyond simple win/loss predictions to provide a rigorous, probabilistic breakdown of the tournament's landscape using applied statistics.

## 🔬 Methodology & Core Engine
The core simulation engine is built in Python (via Google Colab) and executes a 10,000-iteration Monte Carlo simulation. Match outcomes are not treated as deterministic; instead, they are modeled using a **Bivariate Poisson Distribution**.

* **Team Strengths:** A dynamic blend of historical performance decay (SPI-style) and current squad market values (Transfermarkt).
* **Dixon-Coles Adjustment:** Applied a deflationary parameter ($\rho = 0.05$) to correct the standard Poisson model's tendency to underestimate low-scoring draws (0-0, 1-1).
* **Automated CI/CD:** The Colab notebook writes the simulation artifacts (CSVs) directly to this repository using the GitHub API, acting as a lightweight data pipeline that triggers instant front-end updates.

## 📊 The Front-End Experience
The user interface is built with **Streamlit** and **Plotly**, designed as a Business Intelligence dashboard rather than a betting tool. 
* **Survival Analytics:** Visualizes the probability decay of teams advancing through the knockout stages using dynamic thresholding.
* **Group Volatility Index:** Measures the entropy of each group to mathematically identify the "Group of Death."
* **Deterministic Lab:** Allows users to interactively tweak the model's fundamental laws (e.g., Tactical Focus, Underdog Chaos Factor, Regional Boosts) and observe real-time hierarchical shifts.
* **Match Profiler:** Renders the exact probability matrix for any theoretical matchup via a Poisson heatmap.

## 🛠️ Tech Stack
* **Backend / Data Pipeline:** Python, Pandas, NumPy, SciPy (Stats)
* **Frontend:** Streamlit, Plotly
* **Deployment:** Streamlit Community Cloud
