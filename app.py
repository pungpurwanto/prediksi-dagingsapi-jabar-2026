import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. SETTING HALAMAN
st.set_page_config(
    page_title="Forecasting Konsumsi Daging Sapi Jabar 2026",
    page_icon="🥩",
    layout="wide"
)

# 2. FUNGSI LOAD MODEL & DATA
@st.cache_resource
def load_model_assets():
    # Pastikan nama file ini sesuai dengan hasil export script training Anda
    model = joblib.load('model_beef_final_2026.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

@st.cache_data
def load_raw_data():
    # Load data untuk keperluan visualisasi historis
    df = pd.read_csv('daginsapi_2018-2025.csv', sep=';', skiprows=1)
    df.columns = ['Kabupaten_Kota', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    # Pembersihan format angka desimal Indonesia
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

# Memanggil fungsi load
try:
    rf_model, label_encoder = load_model_assets()
    df_history = load_raw_data()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()

# 3. SIDEBAR - PARAMETER PREDIKSI
st.sidebar.header("📊 Kontrol Prediksi")
st.sidebar.markdown("Pilih wilayah dan tahun untuk melihat proyeksi konsumsi.")

district_name = st.sidebar.selectbox("Pilih Kabupaten/Kota", label_encoder.classes_)
target_year = st.sidebar.select_slider("Tahun Proyeksi", options=[2025, 2026, 2027], value=2026)

# 4. HEADER UTAMA
st.title("🥩 Prediksi Konsumsi Daging Sapi Jawa Barat")
st.markdown(f"**Dataset Source:** BPS West Java (2018-2025) | **Model:** Random Forest Regressor")

# 5. LOGIKA PREDIKSI & PERHITUNGAN
district_id = label_encoder.transform([district_name])[0]
input_features = pd.DataFrame({'Tahun': [target_year], 'District_ID': [district_id]})
prediction_value = rf_model.predict(input_features)[0]

# Ambil data tahun 2025 untuk perbandingan (Delta)
val_2025 = df_history.loc[df_history['Kabupaten_Kota'] == district_name, '2025'].values[0]
percent_change = ((prediction_value - val_2025) / val_2025) * 100

# 6. TAMPILAN METRIK UTAMA
st.write("---")
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Wilayah Terpilih", district_name)
with m_col2:
    st.metric(f"Prediksi Tahun {target_year}", f"{prediction_value:.4f} kg", f"{percent_change:.2f}% vs 2025")
with m_col3:
    st.metric("Satuan", "kg/kapita/minggu")

# 7. VISUALISASI TREN (LINE CHART)
st.write("### 📈 Tren Konsumsi Historis & Masa Depan")
years_hist = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
values_hist = df_history.loc[df_history['Kabupaten_Kota'] == district_name].iloc[0, 1:].values

fig = go.Figure()
# Data Historis
fig.add_trace(go.Scatter(x=years_hist, y=values_hist, name='Data BPS (Historis)', 
                         line=dict(color='#31333F', width=3)))
# Titik Proyeksi
fig.add_trace(go.Scatter(x=[target_year], y=[prediction_value], name=f'Proyeksi {target_year}',
                         mode='markers', marker=dict(color='red', size=15, symbol='star')))

fig.update_layout(hovermode="x unified", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# 8. PERBANDINGAN TINGKAT PROVINSI (BAR CHART)
st.write("### 🏆 Peringkat Konsumsi Wilayah (2025)")
top_regions = df_history[['Kabupaten_Kota', '2025']].sort_values(by='2025', ascending=True).tail(10)
fig_bar = px.bar(top_regions, x='2025', y='Kabupaten_Kota', orientation='h', 
                 color='2025', color_continuous_scale='Reds')
fig_bar.update_layout(showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# 9. FOOTER TEKNIS
st.divider()
st.caption("Aplikasi ini dikembangkan untuk mendukung kebijakan ketahanan pangan di Jawa Barat. "
           "Validasi statistik menggunakan uji Shapiro-Wilk menunjukkan distribusi data non-normal, "
           "mendukung penggunaan algoritma ensemble Random Forest untuk akurasi yang lebih baik.")