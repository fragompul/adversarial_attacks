import streamlit as st
import pandas as pd
import os

# Global Page Configuration
st.set_page_config(
    page_title="Adversarial Attacks Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the absolute path of the directory where Home.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")
IMG_PATH = os.path.join(BASE_DIR, "assets", "flow_diagram.png")
CSV_PATH = os.path.join(BASE_DIR, "data", "robustness_metrics.csv")

# Function to load local CSS for custom styling
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to apply styles
load_css(CSS_PATH)

# Helper function to load data dynamically
@st.cache_data
def load_kpi_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

# Page Header
st.title("🛡️ Adversarial Attacks in Vision Deep Learning Models")
st.markdown("""
Welcome to the interactive dashboard for my Double Major in Mathematics and Computer Engineering Bachelor's Thesis (TFG). 
This platform explores the fascinating and dangerous world of **Adversarial Machine Learning**. 

Convolutional Neural Networks (CNNs) have achieved superhuman performance in image classification. However, they possess a critical vulnerability: **Adversarial Examples**. These are carefully crafted, imperceptible noises added to an image that force the neural network to make a highly confident, yet completely incorrect prediction.
""")

# Conceptual Diagram
st.subheader("⚙️ The Concept: How it works")
st.info("The math behind the illusion: By calculating the gradient of the loss function with respect to the input pixels, we can push the image across the model's decision boundaries.")

if os.path.exists(IMG_PATH):
    st.image(IMG_PATH, caption="Conceptual Pipeline of an Adversarial Attack", use_container_width=True)
else:
    st.warning("⚠️ Diagram image not found. Please ensure 'flow_diagram.png' is uploaded to 'dashboard/assets/' on GitHub.")

st.divider()

# Global KPIs
st.subheader("📊 Global Project KPIs")

df_metrics = load_kpi_data()

# Calculate dynamic KPIs
if df_metrics is not None:
    # Most robust model (highest average Accuracy excluding Baseline)
    df_attacks = df_metrics[df_metrics['Attack'] != 'Baseline']
    robust_model = df_attacks.groupby('Model')['Accuracy (%)'].mean().idxmax()
    robust_acc = df_attacks.groupby('Model')['Accuracy (%)'].mean().max()
    
    # Most lethal attack (highest Attack Success Rate)
    lethal_attack = df_attacks.groupby('Attack')['ASR (%)'].mean().idxmax()
    lethal_asr = df_attacks.groupby('Attack')['ASR (%)'].mean().max()
    
    # Most stealthy attack (lowest L2_Distance)
    stealth_attack = df_attacks.groupby('Attack')['Avg_L2'].mean().idxmin()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="🏆 Most Robust Model", value=robust_model, delta=f"{robust_acc:.1f}% Avg. Accuracy", delta_color="normal")
    kpi2.metric(label="⚔️ Most Lethal Attack", value=lethal_attack, delta=f"{lethal_asr:.1f}% Avg. Success Rate", delta_color="inverse")
    kpi3.metric(label="👻 Most Stealthy Attack", value=stealth_attack, delta="Lowest L2 Distortion", delta_color="off")
else:
    st.warning("⚠️ Could not load 'robustness_metrics.csv' from the 'data/' folder. KPIs are currently unavailable.")

st.divider()

# Navigation Guide
st.markdown("""
### 🧭 Dashboard Navigation
Use the sidebar on the left to explore the different sections of this research:
* **🎮 Playground**: Upload your own images or use samples to generate live adversarial attacks.
* **📊 Robustness Evaluation**: Interactive charts (Radar, Heatmaps) comparing model resilience.
* **🌌 Latent Space Exploration**: Discover how attacks shift the internal representations using PCA.
* **🕳️ Attractors & Loss**: Visualize the 3D topology of the network's loss landscape.
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #555555; padding-top: 20px;">
    <p style="font-size: 1.1em; font-weight: 600; margin-bottom: 5px;">Developed by Francisco Javier Gómez Pulido</p>
    <p style="font-size: 0.9em; margin-bottom: 15px;">
        <i>Machine Learning Engineer @ IMSE-cnm (CSIC) | Double Major in Mathematics & Computer Science | Master's in Artificial Intelligence</i>
    </p>
    <p style="font-size: 0.9em;">
        <a href="https://linkedin.com/in/frangomezpulido" target="_blank" style="text-decoration: none; color: #1f77b4;">🔗 LinkedIn</a> &nbsp; | &nbsp; 
        <a href="https://github.com/fragompul" target="_blank" style="text-decoration: none; color: #1f77b4;">🐙 GitHub</a> &nbsp; | &nbsp; 
        <a href="mailto:frangomezpulido2002@gmail.com" style="text-decoration: none; color: #1f77b4;">✉️ Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)
