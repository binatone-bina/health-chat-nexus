from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import os
from typing import Optional
import base64
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Prescription Service", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

def preprocess_image(image_bytes: bytes) -> bytes:
    """Preprocess image to improve OCR accuracy"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image format")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Encode back to bytes
        _, buffer = cv2.imencode('.png', thresh)
        return buffer.tobytes()
    
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image_bytes  # Return original if preprocessing fails

async def extract_text_with_gemini(image_bytes: bytes) -> str:
    """Extract text from image using Gemini AI"""
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create prompt for prescription text extraction
        prompt = """
        You are an expert medical text extraction system. Please extract all text from this prescription image.
        
        Focus on:
        - Patient information
        - Doctor information
        - Medication names and dosages
        - Instructions for use
        - Date and other relevant details
        
        Please format the extracted text clearly and maintain the structure as much as possible.
        If you cannot read certain parts, indicate with [UNCLEAR] or [ILLEGIBLE].
        
        Return only the extracted text without any additional commentary.
        """
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        if response.text:
            return response.text.strip()
        else:
            return "No text could be extracted from the image."
            
    except Exception as e:
        logger.error(f"Gemini text extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OCR Prescription Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "service": "OCR Prescription Service"
    }
    
@app.get("/test-gemini")
async def test_gemini():
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Say hello")
        return {"success": True, "response": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/extract-text")
async def extract_prescription_text(file: UploadFile = File(...)):
    """Extract text from uploaded prescription image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large (max 10MB)")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image_bytes = preprocess_image(image_bytes)
        
        # Extract text using Gemini
        extracted_text = await extract_text_with_gemini(processed_image_bytes)
        
        return JSONResponse(content={
            "success": True,
            "extracted_text": extracted_text,
            "filename": file.filename,
            "file_size": len(image_bytes),
            "message": "Text extracted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
