"""
Video Speaker Detection Module

Analyzes video to detect who is speaking based on visual cues:
- Lip movement detection
- Face detection and tracking
- Motion analysis around mouth region
- Syncs with audio transcript timestamps

This helps improve speaker diarization accuracy by combining audio and video.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import mediapipe as mp


class VideoSpeakerDetector:
    """
    Detects speakers in video using facial landmarks and lip movement analysis
    """
    
    def __init__(self, video_path):
        """
        Initialize the video speaker detector
        
        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  # Support up to 5 speakers
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (MediaPipe Face Mesh)
        # Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
        # Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        self.upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        print(f"Video loaded: {self.fps} FPS, {self.duration:.2f}s duration")
    
    def calculate_lip_distance(self, landmarks):
        """
        Calculate the vertical distance between upper and lower lips
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            float: Average vertical distance between lips (normalized)
        """
        # Get upper lip points
        upper_lip_points = [landmarks[idx] for idx in self.upper_lip_indices]
        lower_lip_points = [landmarks[idx] for idx in self.lower_lip_indices]
        
        # Calculate average y-coordinate for upper and lower lips
        upper_avg_y = np.mean([p.y for p in upper_lip_points])
        lower_avg_y = np.mean([p.y for p in lower_lip_points])
        
        # Return vertical distance
        return abs(lower_avg_y - upper_avg_y)
    
    def detect_speaking_per_frame(self):
        """
        Analyze each frame to detect which faces are speaking
        
        Returns:
            list: List of dicts with frame info and speaking indicators
        """
        results = []
        frame_number = 0
        
        print("Analyzing video frames for speaker detection...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_number / self.fps
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_results = self.face_mesh.process(rgb_frame)
            
            frame_data = {
                'frame': frame_number,
                'timestamp': timestamp,
                'faces': []
            }
            
            if face_results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                    # Calculate lip distance
                    lip_distance = self.calculate_lip_distance(
                        face_landmarks.landmark
                    )
                    
                    # Get face bounding box (for position tracking)
                    h, w, _ = frame.shape
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    
                    face_data = {
                        'face_id': face_idx,
                        'lip_distance': lip_distance,
                        'bbox': {
                            'x': int(min(x_coords)),
                            'y': int(min(y_coords)),
                            'width': int(max(x_coords) - min(x_coords)),
                            'height': int(max(y_coords) - min(y_coords))
                        }
                    }
                    
                    frame_data['faces'].append(face_data)
            
            results.append(frame_data)
            frame_number += 1
            
            # Progress indicator
            if frame_number % 100 == 0:
                progress = (frame_number / self.frame_count) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
        
        print("\nVideo analysis complete!")
        self.cap.release()
        return results
    
    def detect_speaking_segments(self, frame_data, threshold_multiplier=1.5):
        """
        Identify speaking segments by analyzing lip movement patterns
        
        Args:
            frame_data: Frame-by-frame analysis data
            threshold_multiplier: Multiplier for detecting significant lip movement
            
        Returns:
            dict: Speaking segments per face
        """
        # Track each face's lip movements
        face_movements = defaultdict(list)
        
        for frame in frame_data:
            for face in frame['faces']:
                face_id = face['face_id']
                face_movements[face_id].append({
                    'timestamp': frame['timestamp'],
                    'lip_distance': face['lip_distance'],
                    'bbox': face['bbox']
                })
        
        # Detect speaking for each face
        speaking_segments = {}
        
        for face_id, movements in face_movements.items():
            # Calculate baseline (mouth at rest)
            lip_distances = [m['lip_distance'] for m in movements]
            baseline = np.percentile(lip_distances, 20)  # 20th percentile as baseline
            threshold = baseline * threshold_multiplier
            
            # Detect segments where lip distance exceeds threshold
            segments = []
            current_segment = None
            
            for i, movement in enumerate(movements):
                is_speaking = movement['lip_distance'] > threshold
                
                if is_speaking:
                    if current_segment is None:
                        # Start new segment
                        current_segment = {
                            'start': movement['timestamp'],
                            'end': movement['timestamp'],
                            'max_lip_distance': movement['lip_distance'],
                            'bbox': movement['bbox']
                        }
                    else:
                        # Continue segment
                        current_segment['end'] = movement['timestamp']
                        current_segment['max_lip_distance'] = max(
                            current_segment['max_lip_distance'],
                            movement['lip_distance']
                        )
                else:
                    if current_segment is not None:
                        # End segment (with minimum duration filter)
                        if current_segment['end'] - current_segment['start'] >= 0.3:
                            segments.append(current_segment)
                        current_segment = None
            
            # Add final segment if exists
            if current_segment is not None:
                if current_segment['end'] - current_segment['start'] >= 0.3:
                    segments.append(current_segment)
            
            speaking_segments[face_id] = {
                'baseline_lip_distance': baseline,
                'threshold': threshold,
                'segments': segments,
                'total_speaking_time': sum(s['end'] - s['start'] for s in segments)
            }
        
        return speaking_segments
    
    def match_with_transcript(self, speaking_segments, transcript_utterances):
        """
        Match video speaking segments with audio transcript utterances
        
        Args:
            speaking_segments: Speaking segments detected from video
            transcript_utterances: Utterances from audio transcription
            
        Returns:
            list: Utterances with corrected speaker assignments
        """
        corrected_utterances = []
        
        for utterance in transcript_utterances:
            audio_start = utterance['start']
            audio_end = utterance['end']
            audio_speaker = utterance.get('speaker', 0)
            
            # Find which video face was speaking during this time
            max_overlap = 0
            matched_face = None
            
            for face_id, data in speaking_segments.items():
                for segment in data['segments']:
                    # Calculate overlap between audio utterance and video segment
                    overlap_start = max(audio_start, segment['start'])
                    overlap_end = min(audio_end, segment['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_face = face_id
            
            # Create corrected utterance
            corrected = utterance.copy()
            if matched_face is not None:
                corrected['video_speaker'] = matched_face
                corrected['original_audio_speaker'] = audio_speaker
                corrected['speaker_match_confidence'] = max_overlap / (audio_end - audio_start)
                corrected['speaker'] = matched_face  # Override with video-based speaker
            else:
                corrected['video_speaker'] = None
                corrected['speaker_match_confidence'] = 0.0
            
            corrected_utterances.append(corrected)
        
        return corrected_utterances
    
    def analyze_and_sync(self, transcript_data, output_path=None):
        """
        Complete pipeline: analyze video and sync with transcript
        
        Args:
            transcript_data: Transcript data with utterances
            output_path: Optional path to save results
            
        Returns:
            dict: Complete analysis with corrected speaker assignments
        """
        # Step 1: Analyze video frames
        print("\nStep 1: Analyzing video frames...")
        frame_data = self.detect_speaking_per_frame()
        
        # Step 2: Detect speaking segments
        print("\nStep 2: Detecting speaking segments...")
        speaking_segments = self.detect_speaking_segments(frame_data)
        
        # Print summary
        print("\nSpeaking Segments Summary:")
        for face_id, data in speaking_segments.items():
            print(f"  Face {face_id}: {len(data['segments'])} segments, "
                  f"{data['total_speaking_time']:.1f}s total")
        
        # Step 3: Match with transcript
        print("\nStep 3: Matching with audio transcript...")
        corrected_utterances = self.match_with_transcript(
            speaking_segments,
            transcript_data.get('utterances', [])
        )
        
        # Create result
        result = {
            'video_path': str(self.video_path),
            'video_fps': self.fps,
            'video_duration': self.duration,
            'speaking_segments': speaking_segments,
            'corrected_utterances': corrected_utterances,
            'original_transcript': transcript_data.get('transcript', ''),
            'correction_summary': self._generate_summary(corrected_utterances)
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {output_path}")
        
        return result
    
    def _generate_summary(self, corrected_utterances):
        """Generate summary of corrections made"""
        total = len(corrected_utterances)
        corrected = sum(1 for u in corrected_utterances if u.get('video_speaker') is not None)
        high_confidence = sum(1 for u in corrected_utterances 
                             if u.get('speaker_match_confidence', 0) > 0.7)
        
        return {
            'total_utterances': total,
            'video_matched': corrected,
            'high_confidence_matches': high_confidence,
            'match_rate': corrected / total if total > 0 else 0
        }


def sync_video_with_transcript(video_path, transcript_path, output_path=None):
    """
    Convenience function to sync video with transcript
    
    Args:
        video_path: Path to video file
        transcript_path: Path to transcript JSON file
        output_path: Optional path to save synced results
        
    Returns:
        dict: Synced transcript with video-based speaker assignments
    """
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Analyze video and sync
    detector = VideoSpeakerDetector(video_path)
    result = detector.analyze_and_sync(transcript_data, output_path)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python video_speaker_sync.py <video_file> <transcript_json> [output_json]")
        print("\nExample:")
        print("  python video_speaker_sync.py podcast.mp4 transcript.json synced_output.json")
        sys.exit(1)
    
    video_file = sys.argv[1]
    transcript_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "synced_transcript.json"
    
    print("=" * 80)
    print("Video-Audio Speaker Synchronization")
    print("=" * 80)
    
    result = sync_video_with_transcript(video_file, transcript_file, output_file)
    
    print("\n" + "=" * 80)
    print("Synchronization Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    summary = result['correction_summary']
    print(f"  Total utterances: {summary['total_utterances']}")
    print(f"  Video-matched: {summary['video_matched']}")
    print(f"  High confidence: {summary['high_confidence_matches']}")
    print(f"  Match rate: {summary['match_rate']:.1%}")
    print(f"\nResults saved to: {output_file}")