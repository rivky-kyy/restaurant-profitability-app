import streamlit as st
import pandas as pd
import pickle

# ======================
# Load Model & Tools
# ======================
@st.cache_resource
def load_model():
    with open("knn_final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, preprocessor, label_encoder

model, preprocessor, label_encoder = load_model()

# ======================
# Judul & Deskripsi
# ======================
st.set_page_config(page_title="Prediksi Profitability Restoran", page_icon="🍽️", layout="centered")
st.title("🍽️ Prediksi Profitability Restoran")
st.write("""
Aplikasi ini memprediksi **Profitability** (Low / Medium / High) berdasarkan:
- **RestaurantID**
- **MenuCategory**
- **Prices**
""")

# ======================
# Form Input
# ======================
with st.form("input_form"):
    restaurant_id = st.text_input("Restaurant ID")
    menu_category = st.text_input("Menu Category")
    price = st.number_input("Price", min_value=0.0, step=0.01)

    submit = st.form_submit_button("Prediksi")

# ======================
# Prediksi
# ======================
if submit:
    if restaurant_id and menu_category and price:
        # Buat DataFrame input
        new_data = pd.DataFrame({
            "RestaurantID": [restaurant_id],
            "MenuCategory": [menu_category],
            "prices": [price]
        })

        # Transformasi data
        new_data_transformed = preprocessor.transform(new_data)

        # Prediksi
        pred_encoded = model.predict(new_data_transformed)[0]
        mapping = {v: k for k, v in label_encoder.items()}
        prediction_label = mapping[pred_encoded]

        # Output
        st.success(f"📊 Prediksi Profitability: **{prediction_label}**")
    else:
        st.warning("⚠️ Mohon isi semua kolom input.")

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("Dibuat untuk Proyek Mata Kuliah Modern Prediction and Machine Learning")
