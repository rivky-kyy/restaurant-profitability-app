import streamlit as st
import pandas as pd
import pickle

# Load pipeline
with open("knn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍽️ Prediksi Profitability Restoran")
st.write("Masukkan detail restoran dan menu untuk memprediksi keuntungan (Low / Medium / High)")

# Input form
restaurant_id = st.text_input("Restaurant ID")
menu_category = st.text_input("Menu Category")
price = st.number_input("Price", min_value=0.0)

if st.button("Prediksi"):
    if restaurant_id and menu_category and price:
        new_data = pd.DataFrame({
            'RestaurantID': [restaurant_id],
            'MenuCategory': [menu_category],
            'Price': [price]
        })
        
        pred = model.predict(new_data)[0]
        st.success(f"📊 Prediksi Profitability: **{pred}**")
    else:
        st.warning("⚠️ Mohon isi semua kolom input.")

