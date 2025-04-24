# Decoding the Detective: NLP Analysis of Sherlock Holmes Adaptations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
[![NLP Pipeline](https://img.shields.io/badge/NLP%20Pipeline-spaCy%7CBERTopic%7CPhi--3-blueviolet)](https://github.com/mahamayashen/Decoding-the-Detective)

**Quantifying thematic alignment between Arthur Conan Doyle's original Sherlock Holmes novels and modern TV adaptations using cutting-edge NLP techniques**

![Project Workflow Diagram](https://via.placeholder.com/800x400.png?text=NLP+Analysis+Workflow) <!-- Add actual diagram later -->

## 📖 Project Overview
This repository contains a comprehensive NLP pipeline to analyze:

- Thematic evolution from original 19th century text to modern adaptations (CBS's Elementary)
- Semantic similarity between Victorian-era literature and contemporary screenwriting
- Linguistic patterns in character dialogue across different eras

### Key Features
| Component              | Technology Stack       | Purpose                              |
|------------------------|------------------------|--------------------------------------|
| Semantic Search        | ChromaDB               | Vector similarity analysis           |
| Topic Modeling         | BERTopic               | Thematic evolution tracking          |
| Answer Generation      | Phi-3-mini-4k-instruct | Context-aware Q&A system             |
| Linguistic Analysis    | spaCy                  | POS tagging & syntactic patterns      |
| Text Processing        | NLTK, Textacy          | Corpus normalization & feature extraction |

## 🚨 System Requirements
- **Minimum**: 8GB RAM, 5GB disk space
- **Recommended**: 
  - 16GB RAM 
  - NVIDIA GPU (4GB+ VRAM) for accelerated inference
  - SSD storage for vector database operations
- **Model Requirements**:
  - Phi-3-mini-4k-instruct: 2.4GB (Q4_K_M quantized)
  - spaCy en_core_web_lg: 500MB

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/mahamayashen/Decoding-the-Detective.git
cd Decoding-the-Detective
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Manual Model Download
```bash
mkdir -p models/phi3
python -c """
from huggingface_hub import hf_hub_download
hf_hub_download(
repo_id='bartowski/Phi-3-mini-4k-instruct-GGUF',
filename='Phi-3-mini-4k-instruct-Q4_K_M.gguf',
local_dir='models/phi3'
)
"""
python -m spacy download en_core_web_lg
```

