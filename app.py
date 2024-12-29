from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import mysql.connector
import bcrypt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from flask_cors import CORS
from joblib import load
from gdown import download
import os
import pandas as pd

def download_if_not_exists(url, output_path):
    if not os.path.exists(output_path):
        print(f"File {output_path} tidak ditemukan, mengunduh...")
        download(url, fuzzy=True, output=output_path)
    else:
        print(f"File {output_path} sudah ada, tidak perlu mengunduh.")

# Unduh model jika belum ada
download_if_not_exists('https://drive.google.com/uc?id=1tKhSyer94RhQXI9MPIkVU12IQoHEqKYW', 'model/encoders_dict.joblib')
download_if_not_exists('https://drive.google.com/uc?id=1BvdVhaXCsdjI2ZqqjnB362yS3AN8Duu_', 'model/scaler_dict.joblib')
download_if_not_exists('https://drive.google.com/file/d/1eghBY-j6MApXNn0XqxXV2OYa1DCz0DHD/view?usp=drive_link', 'model/calibrated_rf_model.joblib')

# Fungsi untuk memuat data dan prediksi risiko
def risk_predict(data):
    encoders = load('model/encoders_dict.joblib')
    scaler = load('model/scaler_dict.joblib')
    loaded_model = load('model/calibrated_rf_model.joblib')

    new_data = pd.DataFrame([data])

    for col in new_data.select_dtypes(include=['object']).columns:
        new_data[col] = encoders[col].transform(new_data[col])

    for col in new_data.drop(columns=['Age_Category', 'Exercise', 'Sex', 'General_Health', 'Smoking_History']).columns:
        new_data[col] = scaler[col].transform(new_data[[col]])

    hasil_prediksi = loaded_model.predict(new_data)
    proba_prediksi = loaded_model.predict_proba(new_data)

    prediksi_label = 'LOW' if hasil_prediksi[0] == 0 else 'HIGH'
    tingkat_kemungkinan = str(int(round(proba_prediksi[0][hasil_prediksi[0]], 2) * 100)) + '%'

    return prediksi_label, tingkat_kemungkinan

# Fungsi untuk memuat model dan tokenizer
def load_model_tokenizer():
    peft_model_id = "Yudsky/lora-Medical-flan-T5-small"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model, peft_model_id)
    return model, tokenizer

# Load model dan tokenizer
model, tokenizer = load_model_tokenizer()

# Menentukan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)

# Fungsi inferensi
def inference(model, tokenizer, input_sent):
    input_ids = tokenizer(input_sent, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids,
                             top_p=0.9,
                             max_length=512,
                             no_repeat_ngram_size=2,
                             num_beams=10,
                             repetition_penalty=2.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fungsi untuk koneksi database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="dtclq.h.filess.io",
            user="KardAI_droveleast",
            password = os.getenv("DATABASE_PASSWORD"),
            database="KardAI_droveleast",
            port=3307
        )
        if connection.is_connected():
            print("Connected to the database")
            return connection
    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Route untuk halaman home
@app.route("/")
def home():
    logged_in = session.get('logged_in', False)
    username = session.get('username', None)
    email = session.get('email', None)  # Pastikan Anda menyimpan email pengguna di sesi saat login
    return render_template("start.html", logged_in=logged_in, username=username, email=email)
    # return render_template("start.html")

# Route untuk prediksi risiko kesehatan
@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        prediksi, kemungkinan = risk_predict(data)
        return jsonify({"prediction": prediksi, "probability": kemungkinan})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route untuk profil
@app.route('/profile')
def profile():
    if 'logged_in' in session:
        username = session.get('username')
        email = session.get('email')  # Tambahkan email atau data lain jika perlu
        return render_template('start.html', username=username, email=email)
    else:
        flash('You must be logged in to view this page', 'danger')
        return redirect(url_for('login'))

# Route untuk halaman sign-up
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute("INSERT INTO users (Username, Email_User, Password_User) VALUES (%s, %s, %s)",
                               (username, email, password))
                connection.commit()
                flash('Sign Up Successful!', 'success')
                return redirect(url_for('login'))
            except mysql.connector.Error as err:
                flash(f'Error: {err}', 'danger')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Failed to connect to the database', 'danger')
    return render_template('signup.html')

# Route untuk halaman login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT * FROM users WHERE Email_User = %s AND Password_User = %s", (email, password))
                user = cursor.fetchone()
                if user:
                    session['logged_in'] = True
                    session['username'] = user[1]
                    flash('Login Successful!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid email or password', 'danger')
            except mysql.connector.Error as err:
                flash(f'Error: {err}', 'danger')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Failed to connect to the database', 'danger')
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()  # Menghapus semua data sesi pengguna
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))


@app.route('/form')
def form():
    return render_template('form.html')

@app.route("/tes")
def tes():
    return render_template("tes.html")

# Route untuk chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Invalid input"}), 400

    input_text = data["input"]
    response = inference(model, tokenizer, input_text)
    print(f"User input: {input_text}, Response: {response}")
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5100)  # Jangan gunakan debug=True di production