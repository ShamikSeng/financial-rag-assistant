# 🧠 RAG PDFBot - Client

This is the Streamlit-based frontend for the RAG PDFBot. It allows users to upload PDF files, ask questions, inspect responses, and download chat history.

---

## Features

- 📄 Upload and chat with multiple PDFs
- 🧠 Choose between LLM providers (Groq, Gemini)
- 🔬 Inspector to test direct vectorstore queries
- 📥 Downloadable chat history

---

## Project Structure

```
client/
├── app.py                      # Main Streamlit entry point
├── state/
│   └── session.py              # Handles session state setup
├── components/
│   ├── chat.py                 # All chat-related UI: input, history, uploads
│   ├── sidebar.py              # All sidebar elements: model selector, upload, inspect
│   └── inspector.py            # Inspector to test vectorstore responses
├── utils/
│   ├── api.py                  # Server communication
│   ├── config.py               # API_URL etc.
│   └── helpers.py              # High-level orchestration (calls to api.py)
```

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/Zlash65/rag-bot-fastapi.git
cd rag-bot-fastapi
```

2. **Create a virtual environment (optional)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
cd client

pip3 install -r requirements.txt
```

---

## ▶️ Usage

Run the app:

```bash
cd rag-bot-fastapi/client

streamlit run app.py
```

Steps:
1. Select a model provider (Groq or Gemini)
2. Enter your API key
3. Choose a model
4. Upload one or more PDFs
5. Click **Submit**
6. Ask questions about the uploaded PDFs in the chat input

---

## Configuration

Set `API_URL` in `client/utils/config.py` to your FastAPI server:
```python
API_URL = "http://127.0.0.1:8000"
```

---

## 🧼 Tools Panel

- **🔄 Reset**: Clears session state and reruns app
- **🧹 Clear Chat**: Clears chat + PDF submission
- **↩️ Undo**: Removes last question/response

---

## 📦 Download Chat History

Chat history is saved in the session state and can be exported as a CSV with the following columns:

| Question | Answer | Model Provider | Model Name | PDF File | Timestamp |
|----------|--------|----------------|------------|---------------------|-----------|
| What is this PDF about? | This PDF explains... | Groq | openai/gpt-oss-120b | file1.pdf, file2.pdf | 2025-07-03 21:00:00 |

---

## ⚠️ Notes

- Ensure the backend (FastAPI) server is running before launching the client.
- Chat and Inspector require PDFs to be uploaded and processed first.
