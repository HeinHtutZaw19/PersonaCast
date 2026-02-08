from __future__ import annotations

import os
import uuid
import json
from pathlib import Path

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

# Your service (you implement this)
from services.speaker_sync_service import VideoSpeakerDetector

bp = Blueprint("speakers", __name__, url_prefix="/api/v1")

# ---- Paths anchored to backend/ (NOT cwd) ----
BASE_DIR = Path(__file__).resolve().parent          # backend/src/routes
BACKEND_DIR = BASE_DIR.parents[1]                   # backend/
DATA_DIR = BACKEND_DIR / "data"

VIDEO_FOLDER = DATA_DIR / "videos"
TRANSCRIPTS_FOLDER = DATA_DIR / "transcripts"

VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(parents=True, exist_ok=True)

@bp.route("/sync", methods=["POST"])
def sync_video_audio():
    """
    Sync video with transcript to improve speaker turns.
    JSON:
      { "video_id": "...", "transcript_id": "..." }
    OR:
      { "video_path": "...", "transcript_data": {...} }
    """
    data = request.get_json(silent=True) or {}

    # Resolve video file
    video_file = None
    if data.get("video_id"):
        matches = list(VIDEO_FOLDER.glob(f"{data['video_id']}.*"))
        if not matches:
            return jsonify({"error": "Video file not found"}), 404
        video_file = matches[0]
    elif data.get("video_path"):
        video_file = Path(data["video_path"])
        if not video_file.exists():
            return jsonify({"error": "Video file not found"}), 404
    else:
        return jsonify({"error": "video_id or video_path is required"}), 400

    # Resolve transcript data
    transcript_data = None
    transcript_id = data.get("transcript_id")
    if transcript_id:
        transcript_file = TRANSCRIPTS_FOLDER / f"{transcript_id}.json"
        if not transcript_file.exists():
            return jsonify({"error": "Transcript not found"}), 404
        transcript_data = json.loads(transcript_file.read_text())
    elif data.get("transcript_data"):
        transcript_data = data["transcript_data"]
    else:
        return jsonify({"error": "transcript_id or transcript_data is required"}), 400

    detector = VideoSpeakerDetector(str(video_file))
    result = detector.analyze_and_sync(transcript_data)

    synced_id = str(uuid.uuid4())
    out_file = TRANSCRIPTS_FOLDER / f"{synced_id}_synced.json"
    out_file.write_text(json.dumps(result, indent=2))

    return jsonify({
        "success": True,
        "synced_transcript_id": synced_id,
        "video_id": data.get("video_id"),
        "original_transcript_id": transcript_id,
        "correction_summary": result.get("correction_summary"),
        "speaking_segments": result.get("speaking_segments"),
        "corrected_utterances": result.get("corrected_utterances"),
    }), 200
