import os
import uuid
from flask import Blueprint, jsonify
from pathlib import Path

DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY', '')
bp = Blueprint("health", __name__, url_prefix="/api/v1")

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "PersonaCast transcription service is running",
        "deepgram_configured": bool(DEEPGRAM_API_KEY)
    })