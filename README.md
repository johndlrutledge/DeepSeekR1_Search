# 🔍 DeepSeek R1 Web Search Assistant

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AI-powered search assistant combining DeepSeek R1 with real-time web search for comprehensive, sourced answers.

![Example Answer](results/example_answer.avif)

</div>

## 📖 Overview

Combines DeepSeek R1 7B LLM with DuckDuckGo search to provide detailed, sourced answers for research and learning.

## ✨ Key Features

- **AI Processing**: DeepSeek R1 7B model for quality responses
- **Web Search**: Real-time DuckDuckGo integration
- **Sources**: Auto-citation with titles, dates, and snippets
- **Optimized**: Runs on consumer hardware

## 🛠️ System Requirements

- Python 3.8+
- CUDA GPU recommended
- 32GB RAM (without GPU)
- 20GB storage
- Windows/Linux/macOS

## 📦 Quick Start

```bash
git clone https://github.com/johndlrutledge/DeepSeekR1_Search.git
cd DeepSeekR1_Search
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python app.py
```

Access at http://localhost:7860 (opens automatically)


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
