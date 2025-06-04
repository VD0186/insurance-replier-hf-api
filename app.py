from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import time

app = Flask(__name__)
# 限制允许的来源（根据实际需求调整）
CORS(app, resources={
    r"/api": {"origins": ["https://vd0186.github.io"]},
    r"/": {"origins": "*"}
})

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HUGGINGFACE_API_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

# 健康检查端点
@app.route("/")
def health_check():
    return jsonify({"status": "API is running", "model": "flan-t5-base"})

@app.route("/api", methods=["POST"])
def reply():
    # 验证请求数据
    if not request.is_json:
        return jsonify({"error": "请求必须为JSON格式"}), 400
    
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    prompt = question + "。請用繁體中文回答，並針對台灣適用的保險方案舉例、建議合適保險公司與保費。"

    try:
        # 重试机制（处理模型加载中的503错误）
        max_retries = 3
        for attempt in range(max_retries):
            res = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json={"inputs": prompt})
            
            if res.status_code == 200:
                generated = res.json()
                if isinstance(generated, list) and len(generated) > 0:
                    return jsonify({"reply": generated[0].get("generated_text", str(generated))})
                return jsonify({"reply": str(generated)})
            
            elif res.status_code == 503 and attempt < max_retries - 1:
                time.sleep(3)  # 等待模型加载
                continue
                
            return jsonify({"error": f"Hugging Face API 错误: {res.status_code}", "details": res.text}), 500
            
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"连接API失败: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # 生产环境应关闭debug模式
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "false").lower() == "true")
