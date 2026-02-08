
import os
import uuid
from flask import request, jsonify, Blueprint
from pathlib import Path


bp = Blueprint("assets", __name__, url_prefix="/api/v1")
# Configuration
UPLOAD_FOLDER = Path('./uploads')
TRANSCRIPTS_FOLDER = Path('./transcripts')
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'flac', 'ogg', 'webm'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/upload', methods=['POST'])
def upload_audio():
    """
    Upload an audio file for later transcription
    Accepts: multipart/form-data with 'audio' file
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Generate unique ID for this upload
        file_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{file_extension}"
        file_path = UPLOAD_FOLDER / filename
        
        # Save the file
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_size": file_size,
            "file_path": str(file_path),
            "message": "File uploaded successfully. Use /transcribe endpoint to transcribe."
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500