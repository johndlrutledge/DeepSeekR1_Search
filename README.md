# 🔍 DeepSeek R1 Web Search Assistant

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent search assistant powered by DeepSeek R1 that combines real-time web search with advanced AI to deliver comprehensive, well-sourced answers to your questions.

![Example Answer](results/example_answer.avif)

</div>

## 📖 Overview

This project leverages the DeepSeek R1 7B parameter language model along with DuckDuckGo web search to create a powerful question-answering system. It provides detailed, contextual responses while citing sources, making it perfect for research, learning, and general information gathering.

## ✨ Key Features

- **Advanced AI Processing**: Utilizes DeepSeek R1 (7B parameter model) for high-quality responses
- **Real-time Web Search**: Integrated DuckDuckGo search for up-to-date information
- **Source Attribution**: Automatic citation of web sources with titles, dates, and snippets
- **Rich Formatting**: Markdown-formatted responses for clear presentation
- **Performance Optimized**:
  - GPU acceleration support
  - Efficient memory management
  - Disk caching for faster responses
- **Modern Interface**:
  - Dark theme design
  - Real-time status updates
  - Mobile-responsive layout
  - Expandable chat history
  - One-click example questions

## 🛠️ System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU (strongly recommended)
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: 20GB free space for model and cache
- **OS**: Windows 10/11, Linux, or macOS

## 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/deepseek-search-assistant.git
cd deepseek-search-assistant
```

2. **Set Up Python Environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Start the Application**
```bash
python app.py
```

2. **Access the Interface**
- The web interface will automatically open in your default browser
- Default address: http://localhost:7860

3. **Example Queries**:
- "What are the latest developments in quantum computing?"
- "Explain the process of photosynthesis"
- "What are the economic impacts of climate change?"

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Close other GPU-intensive applications
   - Reduce batch size in config
   - Try running on CPU mode

2. **Slow Response Times**
   - Check internet connection
   - Clear cache directory
   - Ensure GPU drivers are up to date

3. **Installation Problems**
   - Update pip: `python -m pip install --upgrade pip`
   - Install Visual C++ Build Tools (Windows)
   - Check CUDA compatibility

## 💻 Development

### Setting Up Development Environment

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for new functions
- Write unit tests for new features

## 📄 License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
