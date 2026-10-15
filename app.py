from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = tf.keras.models.load_model("sentiment_classifier.keras")

@app.route("/")
def home():
    return "Sentiment Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # This simple example expects a sequence of numbers.
    sequence = [int(x) for x in text.split()]
    sequence = pad_sequences([sequence], maxlen=100)

    prediction = model.predict(sequence, verbose=0)[0][0]

    result = "Positive" if prediction >= 0.5 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
