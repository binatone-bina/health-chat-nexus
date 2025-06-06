# OCR Prescription Service

A FastAPI-based service for extracting text from prescription images using Google's Gemini AI.

## Setup

1. Install dependencies:

pip install -r requirements.txt


2. Set up environment variables:


3. Run the service:

python main.py

The service will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /extract-text` - Extract text from prescription image

