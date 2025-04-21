import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 12

# Set page config with custom icon
st.set_page_config(
    page_title="OSCC Diagnosis",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🦷 OSCC Diagnostic Tool")
st.markdown("""
    This tool uses microRNA expression data to predict Oral Squamous Cell Carcinoma (OSCC) 
    and provides mechanistic insights using SHAP visualizations. Adjust miRNA expression levels 
    in the sidebar to explore their impact on tumor progression and diagnostic markers.
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/OSCC_data.xlsx')  # 更新为OSCC数据文件
    return df[['miR-21', 'miR-23b', 'miR-99a', 'let-7b', 'miR-126',
              'let-7i', 'miR-145', 'miR-24', 'miR-27a', 'miR-92a',
              'miR-29a', 'miR-425', 'miR-107', 'miR-22', 'let-7a',
              'miR-146a', 'miR-25', 'miR-20a', 'miR-15b', 'miR-484']]

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/OSCC_MODEL.h5')  # 更新为OSCC模型

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for miRNAs
default_values = {
    'miR-21': 15.49, 'miR-23b': 11.92, 'miR-99a': 7.06,
    'let-7b': 13.5, 'miR-126': 13.04, 'let-7i': 9.21,
    'miR-145': 11.5, 'miR-24': 18.67, 'miR-27a': 11.94,
    'miR-92a': 16.95, 'miR-29a': 10.21, 'miR-425': 11.68,
    'miR-107': 8.53, 'miR-22': 9.98, 'let-7a': 12.08,
    'miR-146a': 11.94, 'miR-25': 15.55, 'miR-20a': 12.02,
    'miR-15b': 11.43, 'miR-484': 14.96
}

# Sidebar configuration
st.sidebar.header("🧬 miRNA Expression Inputs")
st.sidebar.markdown("Adjust expression levels of OSCC-related microRNAs:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# Dynamic two-column layout for 20 miRNAs
mirna_features = list(default_values.keys())
mirna_values = {}
cols = st.sidebar.columns(2)

for i, mirna in enumerate(mirna_features):
    with cols[i % 2]:
        mirna_values[mirna] = st.number_input(
            mirna,
            min_value=float(background_data[mirna].min()),
            max_value=float(background_data[mirna].max()),
            value=default_values[mirna],
            step=0.01,
            format="%.2f",
            key=mirna
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([mirna_values])

# Main analysis
if st.button("🔬 Analyze miRNA Impacts", key="calculate"):
    input_df = prepare_input_data()
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    st.header("📈 Diagnostic Prediction")    
    st.metric("OSCC Probability", f"{prediction:.4f}", 
             delta="Positive" if prediction >= 0.5 else "Negative",
             delta_color="inverse")
    
    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Force Plot", "Waterfall Plot", "Decision Plot", "Mechanistic Insights"])
    
    with tab1:
        st.subheader("Feature Impact Visualization")
        explanation = shap.Explanation(
            values=shap_values, 
            base_values=base_value, 
            feature_names=input_df.columns,
            data=input_df.values
        )
        shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab4:
        st.subheader("Mechanistic Insights")
        st.markdown("""
        **Key OSCC-related Pathways:**
        - miR-21: 肿瘤增殖和转移调控
        - miR-146a: 炎症反应和肿瘤微环境
        - let-7家族: 肿瘤抑制和分化调控
        - miR-29a: 细胞外基质重塑
        - miR-125b: 化疗耐药性调节
        """)
        importance_df = pd.DataFrame({'miRNA': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))

# Update documentation
with st.expander("📚 About This OSCC Analysis", expanded=True):
    st.markdown("""
    ### Model Overview
    本深度学习模型分析20个OSCC相关microRNA，涉及：
    - 肿瘤增殖和转移
    - 炎症微环境调控
    - 上皮间质转化(EMT)
    - 化疗耐药机制
    
    ### SHAP解释指南
    1. **力导向图 (Force Plot)**：显示各miRNA对诊断评分的推拉效应
    2. **瀑布图 (Waterfall Plot)**：特征贡献的逐步可视化
    3. **决策图 (Decision Plot)**：累积效应可视化
    4. **机制解析 (Mechanistic Insights)**：结合SHAP值和已知肿瘤生物学机制的分析
    """)

# Footer
st.markdown("---")
st.markdown(f"Developed for Oral Cancer Research | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")