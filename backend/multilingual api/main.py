from fastapi import FastAPI
from googletrans import Translator
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Text(BaseModel):
    text: str
    class Config:
        from_attributes = True

translator = Translator()

@app.post("/english/")
async def english(text: Text):
    translation = await translator.translate(text.text, dest='en')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/hindi/")
async def hindi(text: Text):
    translation = await translator.translate(text.text, src="en", dest='hi')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/punjabi/")
async def punjabi(text: Text):
    translation = await translator.translate(text.text, dest='pa')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/gujarati/")
async def gujarati(text: Text):
    translation = await translator.translate(text.text, dest='gu')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/bengali/")
async def bengali(text: Text):
    translation = await translator.translate(text.text, dest='bn')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/tamil/")
async def tamil(text: Text):
    translation = await translator.translate(text.text, dest='ta')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/telugu/")
async def telugu(text: Text):
    translation = await translator.translate(text.text, dest='te')
    return JSONResponse(content={"Translation": translation.text})