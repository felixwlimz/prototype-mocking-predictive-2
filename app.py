import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(page_title="V Cosmetics Geo-AI", layout="wide")


@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Pastikan nama kolom koordinat konsisten (misal: 'lat' dan 'lon')
    # Jika di CSV namanya 'latitude'/'longitude', kita rename agar seragam
    df = df.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon',
        'Latitude': 'lat',
        'Longitude': 'lon'
    })
    return df


def main():
    st.title("📍 V Cosmetics - Location Intelligence Dashboard")
    st.markdown("Analisis spasial berdasarkan koordinat presisi dari 50 kota.")

    # 1. LOAD DATA
    filepath = 'v_cosmetics_dataset.csv'
    try:
        df = load_data(filepath)
    except Exception as e:
        st.error(f"Gagal memuat CSV: {e}")
        return

    # 2. SIDEBAR FILTER
    st.sidebar.header("🗺️ Filter Wilayah")
    all_cities = sorted(df['kota'].unique())
    selected_cities = st.sidebar.multiselect(
        "Pilih Kota (Analisis 50 Kota)",
        options=all_cities,
        default=all_cities[:10]  # Default tampilkan 10 dulu agar ringan
    )

    filtered_df = df[df['kota'].isin(selected_cities)]

    if filtered_df.empty:
        st.warning("Silakan pilih minimal satu kota di sidebar.")
        return

    # 3. GEOSPATIAL ANALYSIS (PYDECK 3D)
    st.subheader("🚀 Visualisasi Penjualan Real-Time")

    # Layer 3D Bar (Hexagon) untuk melihat kepadatan transaksi
    view_state = pdk.ViewState(
        latitude=filtered_df['lat'].mean(),
        longitude=filtered_df['lon'].mean(),
        zoom=5,
        pitch=45
    )

    layer_3d = pdk.Layer(
        "HexagonLayer",
        filtered_df,
        get_position=["lon", "lat"],
        radius=5000,
        elevation_scale=100,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer_3d],
        initial_view_state=view_state,
        tooltip={"text": "Kepadatan Transaksi di Area Ini"}
    ))

    st.divider()

    # 4. ANALISIS 50 KOTA (RANKING & PREDIKSI)
    col_rank, col_stat = st.columns([2, 1])

    with col_rank:
        st.write("### 📊 Top Performance: 50 Kota")
        city_rank = df.groupby('kota')['unit_terjual'].sum().sort_values(ascending=False).head(50)
        fig = px.bar(city_rank, x=city_rank.values, y=city_rank.index,
                     orientation='h', color=city_rank.values,
                     labels={'x': 'Total Unit Terjual', 'y': 'Kota'},
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    with col_stat:
        st.write("### 🔮 AI Market Insight")
        total_units = filtered_df['unit_terjual'].sum()
        avg_demand = filtered_df['unit_terjual'].mean()

        st.metric("Total Unit Terjual (Filter)", f"{total_units:,}")
        st.metric("Rata-rata Permintaan", f"{avg_demand:.2f} unit")

        # Analisis Kompetisi vs Penjualan
        st.write("---")
        st.write("**Rekomendasi Strategi:**")
        if filtered_df['kompetitor_count'].mean() > 10:
            st.error("⚠️ Kompetisi Tinggi: Fokus pada loyalitas pelanggan.")
        else:
            st.success("✅ Peluang Ekspansi: Kompetisi rendah, tingkatkan stok!")



if __name__ == "__main__":
    main()