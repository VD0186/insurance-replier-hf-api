
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HUGGINGFACE_API_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

@app.route("/api", methods=["POST"])
def reply():
    data = request.get_json()
    question = data.get("question", "")
    prompt = question + "。請用繁體中文回答，並針對台灣適用的保險方案舉例、建議合適保險公司與保費。"

    try:
        res = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json={"inputs": prompt})
        if res.status_code == 200:
            generated = res.json()
            if isinstance(generated, list) and "generated_text" in generated[0]:
                return jsonify({"reply": generated[0]["generated_text"]})
            else:
                return jsonify({"reply": str(generated)})
        else:
            return jsonify({"error": f"Hugging Face API 錯誤：{res.status_code} - {res.text}"}), 500
    except Exception as e:
        return jsonify({"error": f"❌ 連接 Hugging Face 失敗：{str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
