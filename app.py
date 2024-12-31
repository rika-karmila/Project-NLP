from flask import Flask, request, jsonify
import pickle

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan vectorizer
model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/spam_vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return "Selamat datang di Chatbot Deteksi Spam Email! Gunakan endpoint '/chat' untuk mengirim email."

@app.route('/chat', methods=['POST'])
def chat():
    # Ambil input dari permintaan JSON
    data = request.get_json()
    email = data.get("email")
    
    if not email:
        return jsonify({"error": "Input email tidak ditemukan"}), 400
    
    # Proses input dengan model
    email_vectorized = vectorizer.transform([email])
    prediction = model.predict(email_vectorized)[0]

    # Konversi hasil prediksi ke teks
    result = "Spam" if prediction == 1 else "Not Spam"
    
    # Kirim respons sebagai JSON
    return jsonify({"email": email, "prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
