import streamlit as st
import joblib
import pandas as pd

# ==========================
# Import MODEL and ENCODER
# ==========================

model_dt = joblib.load('decision_tree_for_churn_classification.pkl')
encoders = joblib.load('encoders.pkl')

# ==========================
# Minta Inputan
# ==========================

# Title
st.image(
    image='image/bank.jpeg',
    caption='Bank Illustration'
)
st.title('🏦 Bank Churn Prediction')
st.subheader('Supported with Machine Learning System')
st.divider()
st.subheader('Silahkan input data nasabah!')

# Identifikasikan inputan
customer_id = st.text_input(
    label='ID Nasabah',
    placeholder='Contoh: 15634601',
    max_chars=8,
    help='Masukkan 8 angka ID Nasabah'
).strip()
st.write('ID Nasabah :', customer_id)

credit_score = st.number_input(
    label='Nilai Credit Score Nasabah',
    placeholder='Contoh: 500',
    help='Masukkan score credit Nasabah'
)
st.write('Credit Score :', credit_score)

country = st.selectbox(
    label='Negara Asal Nasabah',
    options=('France', 'Spain', 'Germany'),
    placeholder='Pilih Negara',
    help='Negara asal Nasabah'
)
st.write('Negara asal :', country)

gender = st.selectbox(
    label='Jenis Kelamin Nasabah',
    options=('Male', 'Female'),
    placeholder='Jenis Kelamin',
    help='Pilih jenis kelamin Nasabah',
)
st.write('Jenis kelamin :', gender)

age = st.number_input(
    label='Age',
    max_value=200,
    help='Usia dalam tahun'
)
st.write('Usia :', age)

tenure = st.number_input(
    label='Tenure dalam tahun',
    help='Berapa lama menjadi nasabah di Bank ini',
)
st.write('Tenure :', tenure)

balance = st.number_input(
    label='Balance (Saldo Nasabah) hanya angka',
    placeholder='Contoh: 111098.00',
    help='Saldo nasabah di rekening saat ini',
)
st.write('Balance :', balance)

product_number = st.selectbox(
    label='Jenis Produk Nasabah',
    options=(1, 2, 3),
    placeholder='Nomor Product Nasabah',
    help='1 = Rekening, 2 = Credit, 3 = Pinjaman'
)
if product_number == 1 :
    st.write(f'Product Number : {product_number} = Rekening Tabungan')
elif product_number == 2 :
    st.write(f'Product Number : {product_number} = Credit')
elif product_number == 3 :
    st.write(f'Product Number : {product_number} = Pinjaman')

credit_card = st.selectbox(
    label = 'Apakah Nasabah memiliki Kartu Kredit?',
    options=(0, 1),
    placeholder='Credit Card Nasabah?',
    help='0 = Tidak punya kartu kredit, 1 = punya kartu kredit'
)
if credit_card == 0:
    st.write('Nasabah tidak punya kartu kredit')
elif credit_card == 1:
    st.write('Nasabah punya kartu kredit')

active_member = st.selectbox(
    label='Apakah Nasabah Anggota Aktif?',
    options=(0, 1),
    placeholder='Active Member?',
    help='0 = Tidak Aktif, 1 = Aktif'
)
if active_member == 0:
    st.write('Nasabah tidak aktif')
elif active_member == 1:
    st.write('Nasabah aktif')

estimated_salary = st.number_input(
    label='Estimated Salary ($)',
    placeholder='Contoh: 101348.88',
    help='Masukkan estimasi penghasilan nasabah/month'
)
st.write('Estimasi salary nasabah/month :', estimated_salary)

# ===================================
# Main System (Prediction from Input)
# ===================================

if st.button('Predict'):
    data_df = pd.DataFrame([{
    'credit_score': credit_score,
    'country': country,
    'gender': gender,
    'age': age,
    'tenure': tenure,
    'balance': balance,
    'products_number': product_number,
    'credit_card': credit_card,
    'active_member': active_member,
    'estimated_salary': estimated_salary,
    }])
    data_df = pd.DataFrame(data_df)

    # encoding data
    for col, encoder in encoders.items():
        data_df[col] = encoder.transform(data_df[col])

    # prediction
    prediction = model_dt.predict(data_df)
    result = prediction[0]
    st.success(f'Sukses memprediksi dengan hasil: {result}')
    if result == 1:
        st.write("Nasabah berpotensi Churn")
    elif result == 0:
        st.write('Nasabah tidak berpotensi Churn')
    else:
        st.write('Oops. Terdapat masalah dalam prediksi!!!')