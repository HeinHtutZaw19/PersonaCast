#!/usr/bin/env python3
"""
PersonaCast Backend Test Client

Tests all endpoints of the PersonaCast API
"""

import requests
import json
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_upload_audio(audio_file):
    """Test audio upload endpoint"""
    print_section("Testing Audio Upload")
    
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        response = requests.post(
            f"{BASE_URL}/upload/audio",
            files=files
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 200:
        return data['file_id']
    return None


def test_upload_video(video_file):
    """Test video upload endpoint"""
    print_section("Testing Video Upload")
    
    with open(video_file, 'rb') as f:
        files = {'video': f}
        response = requests.post(
            f"{BASE_URL}/upload/video",
            files=files
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    
    if response.status_code == 200:
        return data['video_id']
    return None


def test_transcribe(file_id):
    """Test transcribe endpoint"""
    print_section("Testing Transcription")
    
    response = requests.post(
        f"{BASE_URL}/transcribe",
        json={"file_id": file_id}
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200:
        print(f"Transcript ID: {data['transcript_id']}")
        print(f"Transcript Preview: {data['transcript'][:200]}...")
        print(f"Utterances: {len(data['utterances'])}")
        
        # Show first few utterances
        print("\nFirst 3 Utterances:")
        for i, utt in enumerate(data['utterances'][:3]):
            print(f"  {i+1}. Speaker {utt['speaker']}: {utt['transcript'][:80]}...")
        
        return data['transcript_id']
    else:
        print(f"Error: {json.dumps(data, indent=2)}")
        return None


def test_transcribe_file(audio_file):
    """Test upload and transcribe in one step"""
    print_section("Testing Upload + Transcribe (One Step)")
    
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        response = requests.post(
            f"{BASE_URL}/transcribe/file",
            files=files
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200:
        print(f"Transcript ID: {data['transcript_id']}")
        print(f"File ID: {data['file_id']}")
        print(f"Transcript Preview: {data['transcript'][:200]}...")
        return data['transcript_id']
    else:
        print(f"Error: {json.dumps(data, indent=2)}")
        return None


def test_list_transcripts():
    """Test list transcripts endpoint"""
    print_section("Testing List Transcripts")
    
    response = requests.get(f"{BASE_URL}/transcripts")
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"Total Transcripts: {data['count']}")
    for transcript in data['transcripts'][:5]:
        print(f"\n  ID: {transcript['transcript_id']}")
        print(f"  Preview: {transcript['preview']}")


def test_get_transcript(transcript_id):
    """Test get specific transcript"""
    print_section("Testing Get Transcript")
    
    response = requests.get(f"{BASE_URL}/transcripts/{transcript_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Transcript ID: {data['transcript_id']}")
        print(f"Utterances: {len(data['utterances'])}")
        print(f"Transcript: {data['transcript'][:300]}...")
    else:
        print(f"Error: {response.json()}")


def test_sync_speakers(video_id, transcript_id):
    """Test video-audio sync endpoint"""
    print_section("Testing Video-Audio Speaker Sync")
    print("This may take a few minutes...")
    
    response = requests.post(
        f"{BASE_URL}/speakers/sync",
        json={
            "video_id": video_id,
            "transcript_id": transcript_id
        }
    )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        summary = data['correction_summary']
        
        print(f"\nSync Complete!")
        print(f"\nCorrection Summary:")
        print(f"  Total Utterances: {summary['total_utterances']}")
        print(f"  Video Matched: {summary['video_matched']}")
        print(f"  High Confidence: {summary['high_confidence_matches']}")
        print(f"  Match Rate: {summary['match_rate']:.1%}")
        
        # Show corrected utterances
        print(f"\nSample Corrected Utterances:")
        for i, utt in enumerate(data['corrected_utterances'][:3]):
            print(f"\n  {i+1}. [{utt['start']:.1f}s - {utt['end']:.1f}s]")
            print(f"     Audio Speaker (original): {utt.get('original_audio_speaker')}")
            print(f"     Video Speaker (corrected): {utt.get('video_speaker')}")
            print(f"     Confidence: {utt.get('speaker_match_confidence', 0):.1%}")
            print(f"     Text: {utt['transcript'][:60]}...")
        
        return data['synced_transcript_id']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None


def test_sync_complete(video_file, audio_file):
    """Test complete sync workflow (upload + transcribe + sync)"""
    print_section("Testing Complete Sync Workflow")
    print("Uploading video and audio, transcribing, and syncing...")
    print("This may take several minutes...")
    
    with open(video_file, 'rb') as v, open(audio_file, 'rb') as a:
        files = {
            'video': v,
            'audio': a
        }
        response = requests.post(
            f"{BASE_URL}/speakers/sync/complete",
            files=files
        )
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        summary = data['correction_summary']
        
        print(f"\n Complete!")
        print(f"\nSynced Transcript ID: {data['synced_transcript_id']}")
        print(f"Video ID: {data['video_id']}")
        print(f"Audio File ID: {data['audio_file_id']}")
        print(f"\nMatch Rate: {summary['match_rate']:.1%}")
        
        return data['synced_transcript_id']
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
        return None


def run_basic_tests(audio_file):
    """Run basic tests (no video)"""
    print("\n" + "=" * 80)
    print(" PersonaCast Backend - Basic Tests")
    print("=" * 80)
    
    # Test health
    if not test_health():
        print("\n Health check failed! Is the server running?")
        return
    
    # Test upload
    file_id = test_upload_audio(audio_file)
    if not file_id:
        print("\n Audio upload failed!")
        return
    
    # Test transcription
    transcript_id = test_transcribe(file_id)
    if not transcript_id:
        print("\n Transcription failed!")
        return
    
    # Test list and get
    test_list_transcripts()
    test_get_transcript(transcript_id)
    
    print("\n" + "=" * 80)
    print("  Basic Tests Complete!")
    print("=" * 80)


def run_full_tests(video_file, audio_file):
    """Run full tests including video sync"""
    print("\n" + "=" * 80)
    print(" PersonaCast Backend - Full Tests (with Video Sync)")
    print("=" * 80)
    
    # Test health
    if not test_health():
        print("\n Health check failed! Is the server running?")
        return
    
    # Test video upload
    video_id = test_upload_video(video_file)
    if not video_id:
        print("\n Video upload failed!")
        return
    
    # Test audio upload and transcription
    file_id = test_upload_audio(audio_file)
    transcript_id = test_transcribe(file_id) if file_id else None
    
    if not transcript_id:
        print("\n Transcription failed!")
        return
    
    # Test video-audio sync
    synced_id = test_sync_speakers(video_id, transcript_id)
    
    if synced_id:
        print("\n" + "=" * 80)
        print("  Full Tests Complete!")
        print("=" * 80)
    else:
        print("\n Video sync failed!")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  # Basic tests (transcription only)")
        print("  python test_backend.py <audio_file>")
        print()
        print("  # Full tests (with video sync)")
        print("  python test_backend.py <video_file> <audio_file>")
        print()
        print("  # Complete sync workflow (one-step)")
        print("  python test_backend.py <video_file> <audio_file> --complete")
        print()
        print("Examples:")
        print("  python test_backend.py podcast.mp3")
        print("  python test_backend.py podcast.mp4 podcast.mp3")
        print("  python test_backend.py podcast.mp4 podcast.mp3 --complete")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Basic tests
        audio_file = Path(sys.argv[1])
        if not audio_file.exists():
            print(f"File not found: {audio_file}")
            sys.exit(1)
        run_basic_tests(audio_file)
    
    elif len(sys.argv) >= 3:
        video_file = Path(sys.argv[1])
        audio_file = Path(sys.argv[2])
        
        if not video_file.exists():
            print(f"Video file not found: {video_file}")
            sys.exit(1)
        
        if not audio_file.exists():
            print(f"Audio file not found: {audio_file}")
            sys.exit(1)
        
        # Check for --complete flag
        if '--complete' in sys.argv:
            test_sync_complete(video_file, audio_file)
        else:
            run_full_tests(video_file, audio_file)


if __name__ == "__main__":
    main()