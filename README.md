# 🔍 DeepSeek R1 Web Search Assistant

An AI-powered search assistant that combines real-time web search with the DeepSeek R1 language model to provide detailed, sourced answers to your questions.

![Example Answer](results/example_answer.avif)

## 🌟 Features

- Real-time web search integration using DuckDuckGo
- AI-powered answer generation with DeepSeek R1 (7B parameter model)
- Markdown-formatted responses with source citations
- Chat history tracking
- GPU-optimized with memory management
- Disk caching for improved performance
- Dark theme modern UI
- Example questions for quick testing

## 🚀 Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The web interface will automatically open in your default browser.

## 💻 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required packages:
  - transformers >= 4.40.0
  - torch >= 2.2.0
  - gradio >= 4.44.1
  - duckduckgo-search >= 3.1.0
  - accelerate >= 0.29.0
  - munch
  - spaces
  - diskcache

## 🔧 How It Works

1. User submits a question through the web interface
2. Application searches DuckDuckGo for relevant web results
3. Search results are formatted into a context-rich prompt
4. DeepSeek R1 model generates a detailed answer using the provided context
5. Response is displayed with markdown formatting and source citations

## 🎨 Interface Features

- Clean, modern dark theme design
- Real-time search status updates
- Expandable chat history
- Source citations with titles, dates, and snippets
- One-click example questions
- Mobile-responsive layout

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license information here]
