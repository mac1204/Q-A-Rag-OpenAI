# PDF Question Answering App

## Overview
This app allows users to upload a PDF and ask questions about its content. It utilizes LangChain for vector-based text retrieval and OpenAI for generating answers.

## Features
- Upload PDFs for content extraction.
- Ask multiple questions about the content.
- Retrieves accurate answers or returns "Data Not Available" for low-confidence queries.

## Installation

### Prerequisites
- Python 3.8 or higher.
- OpenAI API key.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/mac1204/Q-A-Rag-OpenAI.git
   cd Q-A-Rag-OpenAI
   
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
   
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
   
4. Set your OpenAI API key:
    ```bash
    export OPENAI_API_KEY="your-openai-api-key"  # macOS/Linux
    $env:OPENAI_API_KEY="your-openai-api-key"    # Windows

5. Run the application:
    ```bash
    python app.py
   
6. Open your browser at:
    ```bash
    http://127.0.0.1:5000/
   
## Usage
1. Upload a PDF.
2. Enter questions (one per line).
3. Submit and view the JSON responses.

## Deployment
For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
