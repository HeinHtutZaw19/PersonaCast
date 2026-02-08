import os
import uuid
from flask import request, jsonify, Blueprint
from flask_cors import CORS
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from pathlib import Path
import json

bp = Blueprint("transcripts", __name__, url_prefix="/api/v1")
# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent #backend
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
TRANSCRIPTS_FOLDER = DATA_DIR / "transcripts"
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe an uploaded audio file using Deepgram with speaker diarization
    Expected JSON: {"file_id": "uuid"} or {"file_path": "path/to/file"}
    """
    try:
        if not DEEPGRAM_API_KEY:
            return jsonify({"error": "DEEPGRAM_API_KEY not set in environment"}), 500
        
        data = request.get_json()
        file_id = data.get('file_id')
        file_path = data.get('file_path')
        
        # Determine audio file path
        if file_id:
            # Find file with this ID (any extension)
            matching_files = list(UPLOAD_FOLDER.glob(f"{file_id}.*"))
            if not matching_files:
                return jsonify({"error": "Audio file not found"}), 404
            audio_file = matching_files[0]
        elif file_path:
            audio_file = Path(file_path)
        else:
            return jsonify({"error": "file_id or file_path is required"}), 400
        
        if not audio_file.exists():
            return jsonify({"error": "Audio file not found"}), 404
        
        # Initialize Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        # Read audio file
        with open(audio_file, 'rb') as audio:
            buffer_data = audio.read()
        
        payload: FileSource = {
            "buffer": buffer_data,
        }
        
        # Configure Deepgram options with diarization
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=True,  # Enable speaker diarization
            punctuate=True,
            paragraphs=True,
            utterances=True,
        )
        
        # Transcribe
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        # Extract results
        result = response.to_dict()
        
        # Format the response with speaker information
        transcript_id = str(uuid.uuid4())
        transcription_data = {
            "success": True,
            "transcript_id": transcript_id,
            "file_id": file_id if file_id else None,
            "transcript": result['results']['channels'][0]['alternatives'][0]['transcript'],
            "paragraphs": result['results']['channels'][0]['alternatives'][0].get('paragraphs', {}),
            "utterances": result['results']['channels'][0]['alternatives'][0].get('utterances', []),
            "words": result['results']['channels'][0]['alternatives'][0].get('words', []),
        }
        
        # Save transcription to file
        transcript_file = TRANSCRIPTS_FOLDER / f"{transcript_id}.json"
        with open(transcript_file, 'w') as f:
            json.dump(transcription_data, f, indent=2)
        
        return jsonify(transcription_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/transcribe-file', methods=['POST'])
def transcribe_uploaded_file():
    """
    Upload and transcribe in one request
    Accepts: multipart/form-data with 'audio' file
    """
    try:
        if not DEEPGRAM_API_KEY:
            return jsonify({"error": "DEEPGRAM_API_KEY not set in environment"}), 500
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID
        file_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{file_extension}"
        file_path = UPLOAD_FOLDER / filename
        
        # Save the file
        file.save(file_path)
        
        # Transcribe
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        with open(file_path, 'rb') as audio:
            buffer_data = audio.read()
        
        payload: FileSource = {
            "buffer": buffer_data,
        }
        
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=True,
            punctuate=True,
            paragraphs=True,
            utterances=True,
        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        result = response.to_dict()
        
        transcript_id = str(uuid.uuid4())
        transcription_data = {
            "success": True,
            "transcript_id": transcript_id,
            "file_id": file_id,
            "filename": file.filename,
            "transcript": result['results']['channels'][0]['alternatives'][0]['transcript'],
            "paragraphs": result['results']['channels'][0]['alternatives'][0].get('paragraphs', {}),
            "utterances": result['results']['channels'][0]['alternatives'][0].get('utterances', []),
            "words": result['results']['channels'][0]['alternatives'][0].get('words', []),
        }
        
        # Save transcription
        transcript_file = TRANSCRIPTS_FOLDER / f"{transcript_id}.json"
        with open(transcript_file, 'w') as f:
            json.dump(transcription_data, f, indent=2)
        
        return jsonify(transcription_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/transcript/<transcript_id>', methods=['GET'])
def get_transcript(transcript_id):
    """Get saved transcript by transcript_id"""
    try:
        transcript_file = TRANSCRIPTS_FOLDER / f"{transcript_id}.json"
        
        if not transcript_file.exists():
            return jsonify({"error": "Transcript not found"}), 404
        
        with open(transcript_file, 'r') as f:
            data = json.load(f)
        
        return jsonify(data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/transcripts', methods=['GET'])
def list_transcripts():
    """List all available transcripts"""
    try:
        transcripts = []
        for transcript_file in TRANSCRIPTS_FOLDER.glob('*.json'):
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                transcripts.append({
                    'transcript_id': data.get('transcript_id'),
                    'file_id': data.get('file_id'),
                    'filename': data.get('filename'),
                    'preview': data.get('transcript', '')[:100] + '...' if data.get('transcript') else ''
                })
        
        return jsonify({
            "success": True,
            "count": len(transcripts),
            "transcripts": transcripts
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

