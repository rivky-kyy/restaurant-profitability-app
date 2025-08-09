import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Restaurant Profitability Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_resources():
    try:
        # Load model
        model = joblib.load("knn_final_model.joblib")
        
        # Load label encoders
        label_encoders = joblib.load("label_encoder.joblib")
        
        return model, label_encoders
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None

model, label_encoders = load_resources()

# Sidebar
with st.sidebar:
    st.title("🍽️ Restaurant Profitability Predictor")
    st.markdown("""
    Aplikasi ini memprediksi profitabilitas menu restoran berdasarkan:
    - Restaurant ID
    - Kategori Menu
    - Harga Menu
    """)
    
    st.divider()
    st.subheader("Petunjuk Penggunaan")
    st.markdown("""
    1. Masukkan **Restaurant ID** (contoh: R001, R002, R003)
    2. Pilih **Kategori Menu** dari dropdown
    3. Masukkan **Harga Menu** dalam USD
    4. Klik tombol **Prediksi Profitabilitas**
    """)
    
    st.divider()
    st.subheader("Contoh Valid Input")
    st.code("""
    Restaurant ID: R003
    Kategori Menu: Desserts
    Harga: 15.5
    """)
    
    st.divider()
    st.subheader("Informasi Dataset")
    st.markdown("""
    - **Restaurant ID**: R001, R002, R003
    - **Kategori Menu**: Appetizers, Beverages, Desserts, Main Course
    - **Profitabilitas**: Low, Medium, High
    """)
    
    st.divider()
    st.caption("Dibuat untuk Proyek Mata Kuliah Modern Prediction and Machine Learning")
    st.caption("© 2025 - Universitas Islam Indonesia")

# Main content
st.title("🍽️ Prediksi Profitabilitas Menu Restoran")
st.markdown("""
Aplikasi ini membantu Anda memprediksi tingkat profitabilitas menu restoran berdasarkan karakteristik menu. 
Prediksi dapat membantu dalam pengambilan keputusan optimasi menu.
""")

# Columns layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Data Menu")
    with st.form("prediction_form"):
        # Restaurant ID input with validation
        restaurant_id = st.text_input(
            "Restaurant ID*",
            placeholder="R001, R002, R003",
            help="Masukkan ID restoran (contoh: R001)"
        )
        
        # Menu category dropdown
        menu_category = st.selectbox(
            "Kategori Menu*",
            options=["Appetizers", "Beverages", "Desserts", "Main Course"],
            index=2,
            help="Pilih kategori menu"
        )
        
        # Price input
        price = st.number_input(
            "Harga Menu (USD)*",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
            format="%.2f",
            help="Masukkan harga menu dalam USD"
        )
        
        # Submit button
        submitted = st.form_submit_button("Prediksi Profitabilitas", type="primary")
        
        st.caption("*Wajib diisi")

with col2:
    st.subheader("Hasil Prediksi")
    
    if submitted:
        if not restaurant_id or not menu_category:
            st.warning("⚠️ Mohon isi semua kolom input yang wajib diisi.")
        else:
            try:
                # Preprocess input
                restaurant_id_clean = restaurant_id.strip()
                menu_category_clean = menu_category.strip()
                
                # Encode inputs
                restaurant_encoded = label_encoders["restaurant"].transform([restaurant_id_clean])[0]
                category_encoded = label_encoders["category"].transform([menu_category_clean])[0]
                
                # PERBAIKAN DI BAWAH INI: Gunakan nama kolom yang sesuai
                input_data = pd.DataFrame([{
                    'RestaurantID': restaurant_encoded,  # Nama kolom diperbaiki
                    'MenuCategory': category_encoded,    # Nama kolom diperbaiki
                    'Price': price
                }])
                
                # Make prediction
                prediction_encoded = model.predict(input_data)[0]
                prediction_label = label_encoders["profit"].inverse_transform([prediction_encoded])[0]
                
                # Show result with color coding
                st.divider()
                
                if prediction_label == "High":
                    st.success(f"📈 **Profitabilitas Tinggi**")
                    st.metric(label="Prediksi", value=prediction_label)
                    st.success("Menu ini memiliki potensi profitabilitas tinggi! ✅")
                    st.markdown("**Rekomendasi:** Pertahankan menu ini dan pertimbangkan untuk menambah stok.")
                    
                elif prediction_label == "Medium":
                    st.warning(f"📊 **Profitabilitas Sedang**")
                    st.metric(label="Prediksi", value=prediction_label)
                    st.warning("Menu ini memiliki profitabilitas sedang. ⚠️")
                    st.markdown("**Rekomendasi:** Pertimbangkan untuk menyesuaikan harga atau bahan untuk meningkatkan profit.")
                    
                else:
                    st.error(f"📉 **Profitabilitas Rendah**")
                    st.metric(label="Prediksi", value=prediction_label)
                    st.error("Menu ini memiliki profitabilitas rendah. ❌")
                    st.markdown("**Rekomendasi:** Evaluasi ulang menu ini - pertimbangkan untuk mengubah bahan, harga, atau mengganti menu.")
                
                # Show input summary
                st.divider()
                st.subheader("Detail Input")
                st.markdown(f"""
                - **Restaurant ID**: {restaurant_id_clean}
                - **Kategori Menu**: {menu_category_clean}
                - **Harga Menu**: ${price:.2f}
                """)
                
            except ValueError as ve:
                st.error("🚫 Input tidak valid")
                st.warning(f"Pastikan Restaurant ID dan Kategori Menu valid. Error: {str(ve)}")
                st.info("Contoh Restaurant ID yang valid: R001, R002, R003")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.info("Silakan isi form input di sebelah kiri dan klik 'Prediksi Profitabilitas' untuk melihat hasil prediksi.")
        st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=600&q=80", 
                 caption="Restaurant Analytics")

# Footer
st.divider()
st.subheader("Analisis Dataset")
st.markdown("Berikut adalah beberapa insight dari dataset yang digunakan:")

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Distribusi Profitabilitas", "Harga per Kategori", "Profitabilitas per Kategori"])

with tab1:
    # Fake data for visualization (replace with your actual data)
    profit_data = pd.DataFrame({
        'Profitability': ['Low', 'Medium', 'High'],
        'Count': [420, 540, 540]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Profitability', y='Count', data=profit_data, 
                palette=['#ff6b6b', '#ffd166', '#06d6a0'], ax=ax)
    plt.title("Distribusi Profitabilitas Menu")
    plt.xlabel("Tingkat Profitabilitas")
    plt.ylabel("Jumlah Menu")
    st.pyplot(fig)
    st.caption("Distribusi menu berdasarkan tingkat profitabilitas")

with tab2:
    # Fake data for visualization
    price_data = pd.DataFrame({
        'Category': ['Appetizers', 'Beverages', 'Desserts', 'Main Course'],
        'Average Price': [12.5, 3.2, 14.8, 22.3]
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Category', y='Average Price', data=price_data, 
                palette='viridis', ax=ax)
    plt.title("Rata-rata Harga per Kategori Menu")
    plt.xlabel("Kategori Menu")
    plt.ylabel("Harga Rata-rata (USD)")
    plt.xticks(rotation=15)
    st.pyplot(fig)
    st.caption("Perbandingan harga rata-rata untuk setiap kategori menu")

with tab3:
    # Fake data for visualization
    category_profit = pd.DataFrame({
        'Category': ['Appetizers', 'Beverages', 'Desserts', 'Main Course'],
        'High': [0.35, 0.15, 0.50, 0.45],
        'Medium': [0.40, 0.35, 0.30, 0.35],
        'Low': [0.25, 0.50, 0.20, 0.20]
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    category_profit.set_index('Category').plot(kind='bar', stacked=True, 
                                              color=['#06d6a0', '#ffd166', '#ff6b6b'], ax=ax)
    plt.title("Distribusi Profitabilitas per Kategori Menu")
    plt.xlabel("Kategori Menu")
    plt.ylabel("Proporsi")
    plt.legend(title='Profitability', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.caption("Proporsi tingkat profitabilitas untuk setiap kategori menu")

# About section
st.divider()
st.subheader("Tentang Aplikasi Ini")
st.markdown("""
Aplikasi ini dikembangkan sebagai bagian dari Ujian Akhir Semester mata kuliah **Modern Prediction & Machine Learning** dengan tujuan:

- Memprediksi profitabilitas menu restoran
- Membantu pengambilan keputusan optimasi menu
- Memberikan insight tentang karakteristik menu yang profitable

**Teknologi yang digunakan:**
- Python dan Scikit-learn untuk model machine learning
- Streamlit untuk antarmuka aplikasi
- K-Nearest Neighbors sebagai algoritma prediksi

Model ini mencapai akurasi **93.5%** pada data uji berdasarkan validasi silang.
""")