"""
PersonaCast Backend - Main Application Factory

This is the entry point for the PersonaCast transcription service.
Uses Flask application factory pattern with blueprints.
"""

import os
from flask import Flask
from flask_cors import CORS
from pathlib import Path


def create_app(config=None):
    """
    Application factory for PersonaCast
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        DEEPGRAM_API_KEY=os.getenv('DEEPGRAM_API_KEY', ''),
        MAX_CONTENT_LENGTH=500 * 1024 * 1024,  # 500MB max file size
        UPLOAD_FOLDER=Path('./data/uploads'),
        VIDEO_FOLDER=Path('./data/videos'),
        TRANSCRIPTS_FOLDER=Path('./data/transcripts'),
    )
    
    if config:
        app.config.update(config)
    
    # Ensure data directories exist
    app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
    app.config['VIDEO_FOLDER'].mkdir(parents=True, exist_ok=True)
    app.config['TRANSCRIPTS_FOLDER'].mkdir(parents=True, exist_ok=True)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from src.routes import health, assets, transcripts, speakers
    
    app.register_blueprint(health.bp)
    app.register_blueprint(assets.bp)
    app.register_blueprint(transcripts.bp)
    app.register_blueprint(speakers.bp)
    
    # Log startup info
    @app.before_request
    def startup_info():
        print("=" * 80)
        print("PersonaCast Transcription Service")
        print("=" * 80)
        print()
        print(f"Upload folder: {app.config['UPLOAD_FOLDER'].absolute()}")
        print(f"Video folder: {app.config['VIDEO_FOLDER'].absolute()}")
        print(f"Transcripts folder: {app.config['TRANSCRIPTS_FOLDER'].absolute()}")
        print(f"Deepgram API configured: {bool(app.config['DEEPGRAM_API_KEY'])}")
        print()
        print("Available endpoints:")
        print("  GET  /health              - Health check")
        print("  POST /upload/audio        - Upload audio file")
        print("  POST /upload/video        - Upload video file")
        print("  POST /transcribe          - Transcribe uploaded audio")
        print("  POST /transcribe/file     - Upload and transcribe in one step")
        print("  POST /speakers/sync       - Sync video with audio for speaker detection")
        print("  GET  /transcripts         - List all transcripts")
        print("  GET  /transcripts/<id>    - Get specific transcript")
        print()
        print("=" * 80)
        print()
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)