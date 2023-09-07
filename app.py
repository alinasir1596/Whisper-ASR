import io
import logging
import os
import time
import math
from typing import List
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from scipy.io import wavfile
from scipy import signal
import uvicorn
from faster_whisper import WhisperModel

app = FastAPI()

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

logger = logging.getLogger(__name__)

# Load model weights
def load_whisper_model():
    global model
    logger.info("Loading model...")
    model_size = "large-v2"
    load_start_time = time.time()
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    load_end_time = time.time()
    logger.info("Model loaded in %f seconds.", load_end_time - load_start_time)
    # model_load_time = load_end_time - load_start_time
    # return model, model_load_time
    return model

# model, model_load_time = load_whisper_model()
model = load_whisper_model()

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptResponse(BaseModel):
    """
    A response model for the transcribe endpoint.

    Attributes:
    -----------
    text : str
        The transcribed text.
    segments : List[Segment]
        A list of segments if `segments` is set to True, otherwise None.
    start : float
        The start time of the transcription.
    end : float
        The end time of the transcription.
    model_load_time: float
        Time taken to load the model.
    transcription_time: float
        Time taken to transcribe the audio.
    language: str
        The language of the transcription.
    probability: float
        The probability of the language detection.
    duration: float
        Total duration of the audio file.
    """
    text: str = None
    segments: List[Segment] = None
    start: float = None
    end: float = None
    # model_load_time: float = None
    transcription_time: float = None
    language: str = None
    probability: float = None
    duration: float = None

@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe_endpoint(file: UploadFile = File(...), segments: bool = False):
    """
    Transcribes the uploaded audio file and returns the transcribed text.

    Parameters:
    -----------
    file : UploadFile
        The uploaded audio file.
    segments : bool, optional
        Whether to segment the transcription, defaults to False.

    Returns:
    --------
    TranscriptResponse
        A response object containing the transcribed text and, if `segments` is True, a list of segments.

    Raises:
    -------
    HTTPException
        If the uploaded file is not a valid wav file or exceeds the maximum file size.
    """
    file_size = 0
    if hasattr(file, "file"):  # For in-memory file
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)
    else:  # For on-disk file
        file.file_path.seek(0, os.SEEK_END)
        file_size = file.file_path.tell()
        file.file_path.seek(0)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size too large")

    start_time = time.time()

    # Read the audio file from the uploaded .wav file
    logger.info("Reading the audio file...")
    audio_binary = await file.read()
    logger.info("Audio file read.")

    # Check if the uploaded file is a valid wav file
    try:
        sample_rate, audio_data = wavfile.read(io.BytesIO(audio_binary))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid wav file")

    # Resample the audio to 16 kHz
    if sample_rate != 16000:
        resampled_data = signal.resample_poly(audio_data, 16000, sample_rate)
        audio_data = resampled_data.astype(np.int16)

    # Transcribe the audio
    logger.info("Transcribing the audio...")
    start_transcription = time.time()
    transcription_result, info = model.transcribe(audio_data, beam_size=5)
    end_transcription = time.time()
    transcription_time = end_transcription - start_transcription
    logger.info("Audio transcribed in %f seconds.", transcription_time)

    # Get language and language probability
    language, language_probability = (info.language, info.language_probability)

    # Create response object
    response = TranscriptResponse()

    if segments:
        response.segments = [Segment(start=seg.start, end=seg.end, text=seg.text.strip()) for seg in
                             transcription_result]
        response.text = " ".join([seg.text for seg in response.segments])
        response.start = response.segments[0].start
        response.end = response.segments[-1].end
    else:
        response.text = " ".join([seg.text.strip() for seg in transcription_result])
        if any(transcription_result):
            response.start = transcription_result[0].start
            response.end = transcription_result[-1].end
        else:
            response.start = 0.0
            response.end = round(len(audio_data) / float(sample_rate), 2)

    # response.model_load_time = round(model_load_time, 2)
    response.transcription_time = round(transcription_time, 2)
    response.language = language
    response.probability = round(language_probability, 2)
    response.duration = round(len(audio_data) / float(sample_rate), 2)

    logger.info("Total time taken: %f seconds.", time.time() - start_time)

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
