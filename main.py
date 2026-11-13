import streamlit as st
import joblib
import pickle

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
    label='Customer ID',
    placeholder='Contoh: 15634601',
    max_chars=8,
    help='Masukkan 8 angka ID Nasabah'
).strip()
st.write('ID Nasabah :', customer_id)

credit_score = st.number_input(
    label='Credit Score (Contoh 500)',
    placeholder='Contoh: 500',
    help='Masukkan score credit Nasabah'
)
st.write('Credit Score :', credit_score)

country = st.selectbox(
    label='Country',
    options=('Country', 'France', 'Spain', 'Germany'),
    placeholder='Country',
    help='Negara asal Nasabah'
)
st.write('Negara asal :', country)

gender = st.selectbox(
    label='Gender',
    options=('Gender', 'Male', 'Female'),
    placeholder='Gender',
    help='Pilih jenis kelamin Nasabah',
)
st.write('Jenis kelamin :', gender)

age = st.number_input(
    label='Age',
    max_value=200,
    help='Usia dalam tahun'
)
st.write('Usia :', age),

tenure = st.number_input(
    label='Tenure',
    help='Berapa lama menjadi nasabah di Bank ini',
)
st.write('Tenure :', tenure)

balance = st.number_input(
    label='Balance (Saldo Nasabah)',
    help='Saldo nasabah di rekening saat ini',
)
st.write('Balance :', balance)

product_number = st.selectbox(
    label='Product Number',
    options=('Product Number', 1, 2, 3),
    help='1 = Rekening, 2 = Credit, 3 = Pinjaman'
)
if product_number == 1 :
    st.write(f'Product Number : {product_number} = Rekening Tabungan')
elif product_number == 2 :
    st.write(f'Product Number : {product_number} = Credit')
elif product_number == 3 :
    st.write(f'Product Number : {product_number} = Pinjaman')

credit_card = st.selectbox(
    label = 'Credit Card',
    options=('Credit Card', 0, 1),
    help='0 = Tidak punya kartu kredit, 1 = punya kartu kredit'
)
if credit_card == 0:
    st.write('Nasabah tidak punya kartu kredit')
elif credit_card == 1:
    st.write('Nasabah punya kartu kredit')

