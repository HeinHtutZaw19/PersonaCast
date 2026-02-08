import os
import uuid
import json
from pathlib import Path
from flask import request, jsonify, Blueprint

from src.services.deepgram_service import DeepgramService

bp = Blueprint("transcripts", __name__, url_prefix="/api/v1")

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/src
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
TRANSCRIPTS_FOLDER = DATA_DIR / "transcripts"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "mp4", "m4a", "flac", "ogg", "webm"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_deepgram_service() -> DeepgramService:
    # create per-request (simple) or make a global singleton if you prefer
    return DeepgramService()


@bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Expected JSON: {"file_id": "uuid"} or {"file_path": "path/to/file"}
    """
    try:
        data = request.get_json(force=True) or {}
        file_id = data.get("file_id")
        file_path = data.get("file_path")

        if file_id:
            matches = list(UPLOAD_FOLDER.glob(f"{file_id}.*"))
            if not matches:
                return jsonify({"error": "Audio file not found"}), 404
            audio_file = matches[0]
        elif file_path:
            audio_file = Path(file_path)
        else:
            return jsonify({"error": "file_id or file_path is required"}), 400

        if not audio_file.exists():
            return jsonify({"error": f"Audio file not found: {audio_file}"}), 404

        dg = get_deepgram_service()
        transcription_data = dg.transcribe_file(audio_file, file_id=file_id)
        dg.save_transcript(transcription_data, TRANSCRIPTS_FOLDER)

        return jsonify(transcription_data), 200

    except ValueError as e:
        # e.g., missing DEEPGRAM_API_KEY
        return jsonify({"error": str(e)}), 500
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/transcribe-file", methods=["POST"])
def transcribe_uploaded_file():
    """
    Upload and transcribe in one request
    Accepts: multipart/form-data with 'audio' file
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]

        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

        file_id = str(uuid.uuid4())
        ext = file.filename.rsplit(".", 1)[1].lower()
        saved_name = f"{file_id}.{ext}"
        saved_path = UPLOAD_FOLDER / saved_name

        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        file.save(saved_path)

        dg = get_deepgram_service()
        transcription_data = dg.transcribe_file(saved_path, file_id=file_id, original_filename=file.filename)
        dg.save_transcript(transcription_data, TRANSCRIPTS_FOLDER)

        return jsonify(transcription_data), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/transcript/<transcript_id>", methods=["GET"])
def get_transcript(transcript_id):
    try:
        transcript_file = TRANSCRIPTS_FOLDER / f"{transcript_id}.json"
        if not transcript_file.exists():
            return jsonify({"error": "Transcript not found"}), 404

        with open(transcript_file, "r") as f:
            data = json.load(f)

        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/transcripts", methods=["GET"])
def list_transcripts():
    try:
        transcripts = []
        for transcript_file in TRANSCRIPTS_FOLDER.glob("*.json"):
            with open(transcript_file, "r") as f:
                data = json.load(f)
            transcripts.append(
                {
                    "transcript_id": data.get("transcript_id"),
                    "file_id": data.get("file_id"),
                    "filename": data.get("filename"),
                    "preview": (data.get("transcript", "")[:100] + "...") if data.get("transcript") else "",
                }
            )

        return jsonify({"success": True, "count": len(transcripts), "transcripts": transcripts}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
