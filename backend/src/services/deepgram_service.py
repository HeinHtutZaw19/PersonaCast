# src/services/deepgram_service.py

import os
import uuid
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource


load_dotenv()
class DeepgramService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set in environment")

        self.client = DeepgramClient(self.api_key)

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        file_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        model: str = "nova-2",
        diarize: bool = True,
        smart_format: bool = True,
        punctuate: bool = True,
        paragraphs: bool = True,
        utterances: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe a local audio file with Deepgram and return a normalized response dict."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            buffer_data = f.read()

        payload: FileSource = {"buffer": buffer_data}

        options = PrerecordedOptions(
            model=model,
            smart_format=smart_format,
            diarize=diarize,
            punctuate=punctuate,
            paragraphs=paragraphs,
            utterances=utterances,
        )

        response = self.client.listen.rest.v("1").transcribe_file(payload, options)
        result = response.to_dict()

        alt = result["results"]["channels"][0]["alternatives"][0]

        transcript_id = str(uuid.uuid4())
        transcription_data: Dict[str, Any] = {
            "success": True,
            "transcript_id": transcript_id,
            "file_id": file_id,
            "filename": original_filename,
            "transcript": alt.get("transcript", ""),
            "paragraphs": alt.get("paragraphs", {}),
            "utterances": alt.get("utterances", []),
            "words": alt.get("words", []),
        }

        return transcription_data

    def save_transcript(self, transcript: Dict[str, Any], transcripts_folder: Path) -> Path:
        """Save transcript JSON to transcripts_folder/<transcript_id>.json and return the path."""
        transcripts_folder = Path(transcripts_folder)
        transcripts_folder.mkdir(parents=True, exist_ok=True)

        transcript_id = transcript.get("transcript_id")
        if not transcript_id:
            raise ValueError("transcript_id missing from transcript payload")

        out_path = transcripts_folder / f"{transcript_id}.json"
        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2)

        return out_path


if __name__ == "__main__":
    """
    Example:
      export DEEPGRAM_API_KEY="..."
      python -m src.services.deepgram_service --audio ../../data/uploads/sample.mp3 --outdir ../../data/transcripts
    """

    parser = argparse.ArgumentParser(description="Test DeepgramService locally.")
    parser.add_argument("--audio", required=True, help="Path to audio file (mp3/wav/mp4/m4a...)")
    parser.add_argument(
        "--outdir",
        default="data/transcripts",
        help="Output directory to save transcript JSON (default: data/transcripts)",
    )
    parser.add_argument("--file-id", default=None, help="Optional file_id to include in output JSON")
    parser.add_argument("--model", default="nova-2", help="Deepgram model name (default: nova-2)")
    parser.add_argument("--no-diarize", action="store_true", help="Disable diarization")
    args = parser.parse_args()

    service = DeepgramService()

    audio_path = Path(args.audio)
    outdir = Path(args.outdir)

    transcript = service.transcribe_file(
        audio_path,
        file_id=args.file_id,
        original_filename=audio_path.name,
        model=args.model,
        diarize=not args.no_diarize,
    )

    out_path = service.save_transcript(transcript, outdir)

    print("✅ Transcription complete")
    print("Transcript ID:", transcript["transcript_id"])
    print("Saved to:", out_path)
    print("\nPreview:")
    print(transcript["transcript"][:300] + ("..." if len(transcript["transcript"]) > 300 else ""))
