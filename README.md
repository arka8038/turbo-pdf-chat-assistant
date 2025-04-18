# 🚀 PDF Chat Assistant

Chat with any PDF using **blazing fast GROQ LLMs**, powered by **LangChain**, **ChromaDB**, and **Streamlit**.

[![Watch the demo](demo-thumbnail.png)](./Turbo_PDF_Chat_Assistant.mp4)

## ⚡ Features

- 🧠 Ask questions about any PDF — get instant answers  
- 🚀 Ultra-fast responses using [GROQ's](https://groq.com/) LLM APIs  
- 🧩 Built with **LangChain**, **ChromaDB**, and **HuggingFace Embeddings**  
- 📚 Persistent vector storage for seamless memory  
- 🎛️ Streamlit frontend — clean and responsive


## 🛠️ Tech Stack

| Component | Description |
|----------|-------------|
| 🧠 LLM | GROQ (via LangChain integration) |
| 🗃️ Vector DB | ChromaDB |
| 📄 Embeddings | HuggingFace Transformers |
| 💬 Frontend | Streamlit |
| 🧱 Framework | Python |


## 📦 Installation

```bash
git clone https://github.com/yourusername/turbo-pdf-chat.git
pip install -r requirements.txt
```


## 🚀 Usage

```bash
streamlit run app.py
```

1. Upload any PDF  
2. Ask your question  
3. Get instant, contextual answers  

## 🔐 Environment Variables

Create a `.env` file and add the following:

```
GROQ_API_KEY=your_groq_api_key
```

Make sure your API key has access to the LLMs via [GROQ](https://console.groq.com/).

## 📁 Project Structure

```
├── app.py                 # Streamlit UI
├── utils.py               # Helper functions (PDF parsing, etc.)
├── requirements.txt
├── README.md
└── .env                   # API keys (not committed)
```

## 🧠 How It Works

1. **PDF Parsing**: Breaks your document into readable chunks  
2. **Embeddings**: Converts chunks into vectors using HuggingFace  
3. **Vector Store**: ChromaDB stores & retrieves relevant context  
4. **GROQ LLM**: Answers queries in milliseconds using the context  
5. **Streamlit UI**: Lets you chat with your PDF in real time

## 🤝 Contributing

Pull requests are welcome! Feel free to fork, tweak, or extend the project.

## ⭐️ Star This Repo

If you find this project useful, please consider starring 🌟 it to support the work!

## 📬 Contact

For questions, collabs, or feedback:  
📧 ghosh.arka8038@example.com  
🐦 [@arka512x](https://x.com/arka512x)
