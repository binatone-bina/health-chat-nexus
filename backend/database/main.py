from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from mongoDb import save_doctor, save_patient
from mongoDb import get_patient_by_id, get_doctor_by_id, save_meeting, save_voice
from pymongo import MongoClient
import gridfs
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from bson import ObjectId

# For local audio transcription
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

app = FastAPI()

load_dotenv()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/doctor")
async def add_doctor(request: Request):
    data = await request.json()
    doc_id = save_doctor(data)
    return {"doctor_id": str(doc_id)}

@app.post("/patient")
async def add_patient(request: Request):
    data = await request.json()
    pat_id = save_patient(data)
    return {"patient_id": str(pat_id)}

@app.post("/meeting")
async def add_meeting(request: Request):
    data = await request.json()
    meeting_id = save_meeting(data)
    return {"meeting_id": str(meeting_id)}

@app.delete("/meeting/{meeting_id}")
async def delete_meeting(meeting_id: str):
    deleted_count = delete_meeting(meeting_id)
    if deleted_count:
        return {"detail": "Meeting deleted successfully"}
    return {"detail": "Meeting not found"}, 404

@app.post("/patient/login")
async def login_patient(request: Request):
    data = await request.json()
    patient = get_patient_by_id({"id": data})
    if not patient:
        return {"detail": "Patient not found", 1: 404}
    return {"detail": "Login successful", 1: 200}

@app.get("/doctor/{doctor_id}")
async def get_doctor(doctor_id: str):
    doctor = get_doctor_by_id(doctor_id)
    if doctor:
        return doctor
    return {"detail": "Doctor not found"}, 404

client = MongoClient("mongodb://localhost:27017/")
db = client["health_chat"]
fs = gridfs.GridFS(db)

@app.post("/record")
async def record_voice(audio: UploadFile = File(...)):
    contents = await audio.read()
    file_id = fs.put(contents, filename=audio.filename, content_type=audio.content_type)

    # Save to a temp file for transcription
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(contents)
        temp_audio_path = temp_audio.name

    # Convert webm to wav for SpeechRecognition
    wav_path = temp_audio_path.replace(".webm", ".wav")
    try:
        sound = AudioSegment.from_file(temp_audio_path, format="webm")
        sound.export(wav_path, format="wav")

        # Transcribe using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except Exception as e:
                text = f"Transcription failed: {str(e)}"
    except Exception as e:
        text = f"Audio conversion/transcription failed: {str(e)}"
    finally:
        # Clean up temp files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # Save voice info and transcription using your custom function
    save_voice({
        "file_id": str(file_id),
        "filename": audio.filename,
        "content_type": audio.content_type,
        "transcription": text
    })

    return {"file_id": str(file_id), "transcription": text}

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    file = fs.get(ObjectId(file_id))
    return StreamingResponse(file, media_type=file.content_type)