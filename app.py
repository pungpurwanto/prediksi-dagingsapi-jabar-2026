import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Prediksi Konsumsi Daging Sapi Jabar 2026", layout="wide")

# 2. LOAD MODEL DAN ENCODER
@st.cache_resource
def load_assets():
    model = joblib.load('model_beef_final_2026.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

try:
    rf_model, label_encoder = load_assets()
except:
    st.error("File model_beef_final_2026.pkl atau label_encoder.pkl tidak ditemukan. Silakan jalankan script training terlebih dahulu.")

# 3. LOAD DATA HISTORIS (Untuk Grafik)
@st.cache_data
def load_data():
    df = pd.read_csv('daginsapi_2018-2025.csv', sep=';', skiprows=1)
    df.columns = ['Kabupaten_Kota', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

df_raw = load_data()

# 4. SIDEBAR - INPUT USER
st.sidebar.header("Konfigurasi Prediksi")
st.sidebar.info("Model ini menggunakan data historis 2018-2025 untuk memproyeksikan kebutuhan tahun 2026.")

selected_district = st.sidebar.selectbox("Pilih Kabupaten/Kota:", label_encoder.classes_)
target_year = st.sidebar.slider("Tahun Proyeksi:", 2025, 2027, 2026)

# 5. HEADER UTAMA
st.title("🥩 Dashboard Analisis & Prediksi Konsumsi Daging Sapi")
st.subheader(f"Provinsi Jawa Barat - Fokus Wilayah: {selected_district}")

# 6. LOGIKA PREDIKSI
district_id = label_encoder.transform([selected_district])[0]
input_data = pd.DataFrame({'Tahun': [target_year], 'District_ID': [district_id]})
prediction = rf_model.predict(input_data)[0]

# 7. TAMPILAN METRIK
col1, col2, col3 = st.columns(3)
with col1:
    val_2025 = df_raw.loc[df_raw['Kabupaten_Kota'] == selected_district, '2025'].values[0]
    st.metric(label="Konsumsi Riil 2025", value=f"{val_2025:.4f} kg")

with col2:
    st.metric(label=f"Prediksi Tahun {target_year}", value=f"{prediction:.4f} kg", 
              delta=f"{((prediction-val_2025)/val_2025)*100:.2f}% dari 2025")

with col3:
    avg_jabar = df_raw['2025'].mean()
    st.metric(label="Rata-rata Jabar 2025", value=f"{avg_jabar:.4f} kg")

# 8. VISUALISASI TREN
st.markdown("---")
st.subheader(f"Tren Konsumsi Historis & Proyeksi {target_year}")

# Menyiapkan data untuk grafik
historical_values = df_raw.loc[df_raw['Kabupaten_Kota'] == selected_district].iloc[0, 1:].values
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

fig = go.Figure()
# Garis Historis
fig.add_trace(go.Scatter(x=years, y=historical_values, name="Data Historis (BPS)",
                         line=dict(color='royalblue', width=4)))
# Titik Prediksi
fig.add_trace(go.Scatter(x=[target_year], y=[prediction], name=f"Proyeksi {target_year}",
                         mode='markers', marker=dict(color='red', size=12, symbol='star')))

fig.update_layout(xaxis_title="Tahun", yaxis_title="kg/kapita/minggu",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# 9. PERBANDINGAN ANTAR WILAYAH (TOP 10)
st.markdown("---")
st.subheader("Perbandingan Konsumsi Tahun 2025 antar Wilayah")
top_10 = df_raw[['Kabupaten_Kota', '2025']].sort_values(by='2025', ascending=False).head(10)
fig_bar = px.bar(top_10, x='2025', y='Kabupaten_Kota', orientation='h',
                 title="10 Wilayah dengan Konsumsi Tertinggi 2025",
                 labels={'2025': 'Konsumsi (kg)', 'Kabupaten_Kota': ''},
                 color='2025', color_continuous_scale='Reds')
st.plotly_chart(fig_bar, use_container_width=True)

# 10. FOOTER UNTUK JURNAL
st.info("**Catatan Metodologi:** Prediksi dihasilkan menggunakan algoritma *Random Forest Regressor* yang telah divalidasi dengan uji normalitas Shapiro-Wilk (Statistik: 0.832).")
