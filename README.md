
# insurance-replier-hf-api

使用 Hugging Face Inference API 的 Flask 保險回覆系統。

## 🚀 快速部署步驟

1. 建立 Hugging Face 帳號 → https://huggingface.co
2. 取得 API Token → https://huggingface.co/settings/tokens
3. 把本專案上傳到 GitHub
4. 點此部署 → https://render.com/deploy
5. 在 Render 設定環境變數：`HF_TOKEN`（就是你的 Hugging Face token）

## API 呼叫方式

POST `/api`

```json
{
  "question": "我今年30歲，想買醫療險"
}
```
