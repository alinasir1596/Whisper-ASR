# Faster-Whisper Speech-to-Text API

Faster-Whisper Speech-to-Text API is a Python application that provides an API endpoint for transcribing audio files in real-time. It is built on top of FastAPI and uses the Faster-Whisper large-v2 model for transcription.

## Installation

1. Clone the repository
2. Install the required dependencies by running `pip install -r requirements.txt`
3. Start the server by running `uvicorn main:app --reload` from the project directory. The API will start running on http://localhost:8000.

## Usage

To transcribe an audio file, make a POST request to the `/transcribe` endpoint with the audio file as a `multipart/form-data` request body. The response will contain the transcribed text.

### Request

- **Method:** POST
- **Endpoint:** `/transcribe`
- **Headers:**
  - `Content-Type: multipart/form-data`
- **Body:**
  - `file`: the audio file to transcribe
- **Query parameters:**
  - `segments` (optional): whether to segment the transcription, defaults to `False`

### Response

The response will be a JSON object with the following fields:

- `text`: The transcribed text.
- `segments`: A list of segments if `segments` is set to True, otherwise None.
- `start`: The start time of the transcription.
- `end`: The end time of the transcription.
- `model_load_time`: Time taken to load the model.
- `transcription_time`: Time taken to transcribe the audio.
- `language`: The language of the transcription.
- `probability`: The probability of the language detection.
- `duration`: Total duration of the audio file.

If the uploaded file is not a valid wav file or exceeds the maximum file size, the API will return a `HTTP 400 Bad Request` response.

## Configuration

The following environment variables can be used to configure the application:

- `LOG_LEVEL`: The log level to use (default is `INFO`).
- `MAX_FILE_SIZE_BYTES`: The maximum file size allowed for uploaded files (default is 50MB).

## License

This project is licensed under the terms of the MIT license.
