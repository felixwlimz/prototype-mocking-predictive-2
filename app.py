import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Viva Cosmetics AI Dashboard", layout="wide")


def load_dashboard_data():
    """Simulasi data hasil training untuk keperluan demo dashboard"""
    # Ganti ini dengan hasil 'comparison_df' dari main() Anda
    data = {
        'model': ['XGBoost', 'LightGBM', 'PyTorch DNN', 'Random Forest'],
        'r2': [0.892, 0.885, 0.870, 0.842],
        'rmse': [12.4, 13.1, 14.5, 16.2],
        'mae': [8.2, 8.9, 9.4, 10.1],
        'mape': [5.2, 5.5, 6.1, 7.2]
    }
    return pd.DataFrame(data)


def main():
    st.title("💄 Viva Cosmetics - Sales Predictive AI")
    st.markdown("""
    Dashboard ini menampilkan hasil analisis prediksi penjualan unit produk Viva Cosmetics 
    menggunakan perbandingan model **Traditional ML** dan **Deep Learning (PyTorch)**.
    """)

    # Load data
    df_results = load_dashboard_data()

    # --- ROW 1: KEY METRICS ---
    st.subheader("🏆 Model Performance Overview")
    col1, col2, col3, col4 = st.columns(4)

    best_model = df_results.iloc[0]
    col1.metric("Best Model", best_model['model'])
    col2.metric("Highest R² Score", f"{best_model['r2']:.4f}")
    col3.metric("Avg RMSE", f"{df_results['rmse'].mean():.2f}")
    col4.metric("Avg MAPE", f"{df_results['mape'].mean():.2f}%")

    st.divider()

    # --- ROW 2: VISUALIZATIONS ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.write("### Model Comparison (R² Score)")
        fig, ax = plt.subplots()
        sns.barplot(x='r2', y='model', data=df_results, palette='viridis', ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    with col_right:
        st.write("### Error Metrics (Lower is Better)")
        fig, ax = plt.subplots()
        df_melted = df_results.melt(id_vars='model', value_vars=['rmse', 'mae'])
        sns.barplot(x='value', y='model', hue='variable', data=df_melted, ax=ax)
        st.pyplot(fig)

    # --- ROW 3: INTERACTIVE PREDICTION ---
    st.divider()
    st.subheader("🔮 Interactive Sales Predictor")

    with st.expander("Klik untuk melakukan simulasi input data"):
        c1, c2, c3 = st.columns(3)
        input_data = {}

        with c1:
            input_data['harga'] = st.number_input("Harga Produk", min_value=0, value=25000)
            input_data['kategori'] = st.selectbox("Kategori", ["Skincare", "Bodycare", "Makeup"])
        with c2:
            input_data['promo'] = st.slider("Promo Aktif (%)", 0, 100, 10)
            input_data['iklan'] = st.selectbox("Intensitas Iklan", ["Low", "Medium", "High"])
        with c3:
            input_data['stok'] = st.number_input("Stok Tersedia", min_value=0, value=500)
            input_data['rating'] = st.slider("Rating Produk", 1.0, 5.0, 4.5)

        if st.button("Predict Unit Terjual"):
            # Di sini Anda akan memanggil model.predict() atau model(input_tensor)
            # Untuk demo, kita gunakan angka acak
            prediction = np.random.randint(100, 1000)
            st.success(f"Estimasi Unit Terjual: **{prediction} unit**")

    # --- ROW 4: RAW DATA ---
    st.divider()
    st.write("### Detailed Metrics Table")
    st.dataframe(df_results.style.highlight_max(axis=0, subset=['r2']).highlight_min(axis=0, subset=['rmse', 'mae']))


if __name__ == "__main__":
    main()