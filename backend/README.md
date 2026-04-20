# GLA Admission Chatbot — RAG Based

Groq + LangChain + FAISS se bana hua RAG chatbot jo GLA University ke
PDF brochures se seedha jawab deta hai.

## Setup (3 steps)

### 1. Install karo
```bash
pip install -r requirements.txt
```

### 2. API Key set karo
```bash
cp .env.example .env
# .env mein apni Groq API key daalo
# Key milegi: https://console.groq.com
```

### 3. PDFs daalo aur ingest karo
```bash
# Apne GLA brochures data/ folder mein daalo
python scripts/ingest.py
```

### 4. Chatbot chalao
```bash
streamlit run app.py
```

---

## File Structure

```
gla_chatbot/
├── data/              ← PDF brochures yahan
├── vector_store/      ← auto-generate (touch mat karna)
├── src/
│   ├── ocr_loader.py  ← PDF → text
│   ├── chunker.py     ← text → chunks
│   ├── embedder.py    ← chunks → FAISS
│   ├── retriever.py   ← query → relevant chunks
│   ├── groq_llm.py    ← Groq API
│   └── chatbot.py     ← sab ka combination
├── scripts/
│   └── ingest.py      ← ek baar chalao
├── app.py             ← Streamlit UI
├── .env               ← API key (git mein mat daalna!)
└── requirements.txt
```

## Troubleshooting

**"GROQ_API_KEY nahi mili"** → `.env` file check karo

**"Vector store nahi mila"** → `python scripts/ingest.py` chalao

**Scanned PDF kaam nahi kar rahi** → Tesseract install karo:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Mac: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`