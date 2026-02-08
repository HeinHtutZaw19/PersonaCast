"""
Production Video Speaker Detector - Part 2
Complete implementation with InsightFace and Akamai database
"""

from src.utils.face_recognition import (
    AkamaiDatabaseManager,
    ProductionFaceRecognitionTracker
)
from dotenv import load_dotenv
import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import hashlib
import mediapipe as mp


load_dotenv()

class ProductionVideoSpeakerDetector:
    """
    Production-grade multi-shot video speaker detector
    - InsightFace (ArcFace) embeddings
    - Akamai database persistence
    - Lip movement detection
    """
    
    def __init__(self, video_path, db_config, model_name='buffalo_l', device='cpu'):
        """
        Initialize detector
        
        Args:
            video_path: Path to video file
            db_config: Database configuration dict
            model_name: InsightFace model name
            device: 'cpu' or 'cuda'
        """
        self.video_path = Path(video_path)
        self.video_uuid = hashlib.sha256(str(video_path).encode()).hexdigest()[:16]
        
        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count > 0 else 0.0
        
        # Initialize database and face recognition
        print(f"\n{'='*80}")
        print(f"Initializing Production Speaker Detector")
        print(f"{'='*80}")
        print(f"Video: {video_path}")
        print(f"Duration: {self.duration:.2f}s, FPS: {self.fps:.2f}")
        
        self.db_manager = AkamaiDatabaseManager(db_config)
        self.face_tracker = ProductionFaceRecognitionTracker(
            db_manager=self.db_manager,
            model_name=model_name,
            device=device
        )
    
    def detect_faces_and_identify_per_frame(self):
        """
        Process video frame by frame:
        - Detect faces with InsightFace
        - Extract ArcFace embeddings
        - Identify/register persons in database
        - Track lip movements (if MediaPipe available)
        
        Returns:
            list: Frame-by-frame analysis
        """
        results = []
        frame_number = 0
        
        print(f"\n{'='*80}")
        print("Analyzing Video with Production Face Recognition")
        print(f"{'='*80}\n")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_number / self.fps
            
            frame_data = {
                'frame': frame_number,
                'timestamp': float(timestamp),
                'faces': []
            }
            
            # Extract face embeddings with InsightFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_tracker.app.get(rgb_frame)
            
            for face_idx, face in enumerate(faces):
                # Get embedding (512-dim ArcFace)
                embedding = face.embedding
                quality = float(face.det_score) if hasattr(face, 'det_score') else 0.9
                
                # Identify or register person in database
                person_id = self.face_tracker.identify_or_register_person(
                    embedding=embedding,
                    video_id=self.video_uuid,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    quality_score=quality
                )
                
                # Get bounding box
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                
                # Try to get lip movement (requires MediaPipe)
                lip_distance = 0.0
                if self.face_tracker.face_landmarker is not None:
                    # Extract face region and get landmarks
                    try:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            mp_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            )
                            timestamp_ms = int(frame_number * 1000.0 / self.fps)
                            det = self.face_tracker.face_landmarker.detect_for_video(
                                mp_image, timestamp_ms
                            )
                            
                            if det.face_landmarks:
                                lip_distance = self.face_tracker.calculate_lip_distance(
                                    det.face_landmarks[0]
                                )
                    except Exception as e:
                        print(f"  ⚠ Lip detection failed for frame {frame_number}: {e}")
                
                face_data = {
                    'face_idx': face_idx,
                    'person_id': int(person_id),
                    'embedding_quality': float(quality),
                    'lip_distance': float(lip_distance),
                    'bbox': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                }
                
                frame_data['faces'].append(face_data)
            
            results.append(frame_data)
            frame_number += 1
            
            # Progress indicator
            if frame_number % 100 == 0:
                progress = (frame_number / self.frame_count) * 100 if self.frame_count > 0 else 0
                person_stats = self.db_manager.get_person_statistics()
                print(f"Progress: {progress:.1f}% | Persons in DB: {len(person_stats)}", end='\r')
        
        print(f"\n\n✓ Video analysis complete!")
        self.cap.release()
        
        # Get final statistics
        person_stats = self.db_manager.get_person_statistics()
        print(f"Total unique persons in database: {len(person_stats)}")
        for stat in person_stats[:5]:  # Show top 5
            print(f"  Person {stat['person_id']}: {stat['embedding_count']} embeddings, "
                  f"{stat['video_count']} videos")
        
        return results
    
    def detect_speaking_segments(self, frame_data, threshold_multiplier=1.5):
        """
        Detect speaking segments per person based on lip movement
        
        Args:
            frame_data: Frame-by-frame analysis
            threshold_multiplier: Sensitivity for lip movement
            
        Returns:
            dict: Speaking segments per person_id
        """
        # Collect lip movements per person
        person_movements = defaultdict(list)
        
        for frame in frame_data:
            for face in frame['faces']:
                person_id = face['person_id']
                person_movements[person_id].append({
                    'timestamp': frame['timestamp'],
                    'lip_distance': face['lip_distance'],
                    'quality': face['embedding_quality'],
                    'bbox': face['bbox']
                })
        
        speaking_segments = {}
        
        print(f"\n{'='*80}")
        print("Detecting Speaking Segments")
        print(f"{'='*80}\n")
        
        for person_id, movements in person_movements.items():
            if not movements:
                continue
            
            # Calculate baseline lip distance (mouth at rest)
            lip_distances = [m['lip_distance'] for m in movements]
            baseline = float(np.percentile(lip_distances, 20))
            threshold = baseline * threshold_multiplier
            
            # Detect segments where lip movement exceeds threshold
            segments = []
            current = None
            
            for m in movements:
                is_speaking = m['lip_distance'] > threshold
                
                if is_speaking:
                    if current is None:
                        current = {
                            'start': m['timestamp'],
                            'end': m['timestamp'],
                            'max_lip_distance': m['lip_distance'],
                            'avg_quality': m['quality'],
                            'bbox': m['bbox']
                        }
                    else:
                        current['end'] = m['timestamp']
                        current['max_lip_distance'] = max(
                            current['max_lip_distance'], 
                            m['lip_distance']
                        )
                else:
                    if current is not None:
                        # End segment (minimum 0.3s duration)
                        if current['end'] - current['start'] >= 0.3:
                            segments.append(current)
                        current = None
            
            # Add final segment
            if current is not None and current['end'] - current['start'] >= 0.3:
                segments.append(current)
            
            speaking_segments[person_id] = {
                'baseline_lip_distance': baseline,
                'threshold': threshold,
                'segments': segments,
                'total_speaking_time': sum(s['end'] - s['start'] for s in segments),
                'appearance_count': len(movements)
            }
            
            print(f"Person {person_id}: {len(segments)} speaking segments, "
                  f"{speaking_segments[person_id]['total_speaking_time']:.1f}s total")
        
        return speaking_segments
    
    def match_with_transcript(self, speaking_segments, transcript_utterances):
        """
        Match video speaking segments with audio transcript
        
        Args:
            speaking_segments: Speaking segments per person
            transcript_utterances: Audio transcript utterances
            
        Returns:
            list: Corrected utterances with person IDs
        """
        corrected_utterances = []
        
        print(f"\n{'='*80}")
        print("Matching Speakers with Transcript")
        print(f"{'='*80}\n")
        
        for utt in transcript_utterances:
            audio_start = float(utt['start'])
            audio_end = float(utt['end'])
            audio_speaker = utt.get('speaker', 0)
            duration = max(1e-6, audio_end - audio_start)
            
            # Find person with maximum overlap
            max_overlap = 0.0
            matched_person = None
            
            for person_id, data in speaking_segments.items():
                for segment in data['segments']:
                    overlap_start = max(audio_start, segment['start'])
                    overlap_end = min(audio_end, segment['end'])
                    overlap = max(0.0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_person = person_id
            
            # Create corrected utterance
            corrected = dict(utt)
            if matched_person is not None:
                corrected['video_speaker'] = matched_person
                corrected['original_audio_speaker'] = audio_speaker
                corrected['speaker_match_confidence'] = float(max_overlap / duration)
                corrected['speaker'] = matched_person
            else:
                corrected['video_speaker'] = None
                corrected['speaker_match_confidence'] = 0.0
            
            corrected_utterances.append(corrected)
        
        # Print summary
        matched = sum(1 for u in corrected_utterances if u.get('video_speaker') is not None)
        print(f"Matched {matched}/{len(corrected_utterances)} utterances to video speakers")
        
        return corrected_utterances
    
    def analyze_and_sync(self, transcript_data, output_path=None):
        """
        Complete pipeline with database persistence
        
        Args:
            transcript_data: Transcript with utterances
            output_path: Optional output file path
            
        Returns:
            dict: Analysis results
        """
        # Step 1: Detect faces and identify persons
        frame_data = self.detect_faces_and_identify_per_frame()
        
        # Step 2: Detect speaking segments
        speaking_segments = self.detect_speaking_segments(frame_data)
        
        # Step 3: Match with transcript
        corrected_utterances = self.match_with_transcript(
            speaking_segments,
            transcript_data.get('utterances', [])
        )
        
        # Step 4: Save to database
        print(f"\n{'='*80}")
        print("Saving Results to Database")
        print(f"{'='*80}\n")
        
        # Save video metadata
        person_stats = self.db_manager.get_person_statistics()
        self.db_manager.save_video_metadata(
            video_uuid=self.video_uuid,
            video_path=str(self.video_path),
            duration=self.duration,
            fps=self.fps,
            total_persons=len(person_stats),
            metadata={'speaking_segments': len(speaking_segments)}
        )
        
        # Save utterances
        self.db_manager.save_utterances(self.video_uuid, corrected_utterances)
        
        print("✓ Results saved to database")
        
        # Generate summary
        total = len(corrected_utterances)
        matched = sum(1 for u in corrected_utterances if u.get('video_speaker') is not None)
        high_conf = sum(1 for u in corrected_utterances if u.get('speaker_match_confidence', 0) > 0.7)
        
        speaker_counts = defaultdict(int)
        for u in corrected_utterances:
            speaker_id = u.get('speaker', u.get('video_speaker', 'unknown'))
            speaker_counts[str(speaker_id)] += 1
        
        result = {
            'video_path': str(self.video_path),
            'video_uuid': self.video_uuid,
            'video_fps': float(self.fps),
            'video_duration': float(self.duration),
            'total_unique_persons': len(person_stats),
            'person_statistics': person_stats,
            'speaking_segments': speaking_segments,
            'corrected_utterances': corrected_utterances,
            'original_transcript': transcript_data.get('transcript', ''),
            'correction_summary': {
                'total_utterances': total,
                'video_matched': matched,
                'high_confidence_matches': high_conf,
                'match_rate': float(matched / total) if total > 0 else 0.0,
                'speaker_distribution': dict(speaker_counts)
            }
        }
        
        # Save to file if requested
        if output_path:
            self._save_results(result, output_path)
        
        return result
    
    def _save_results(self, result, output_path):
        """Save results to JSON file"""
        # Convert numpy types for JSON
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        result_json = convert_types(result)
        
        with open(output_path, 'w') as f:
            json.dump(result_json, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}")


def sync_video_with_transcript_production(video_path, transcript_path, db_config,
                                         output_path=None, model_name='buffalo_l',
                                         device='cpu'):
    """
    Production sync with database persistence
    
    Args:
        video_path: Path to video
        transcript_path: Path to transcript JSON
        db_config: Database configuration dict
        output_path: Optional output file
        model_name: InsightFace model
        device: 'cpu' or 'cuda'
        
    Returns:
        dict: Analysis results
    """
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Create detector
    detector = ProductionVideoSpeakerDetector(
        video_path=video_path,
        db_config=db_config,
        model_name=model_name,
        device=device
    )
    
    # Analyze and sync
    result = detector.analyze_and_sync(transcript_data, output_path)
    
    return result


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python production_speaker_detector.py <video> <transcript> [output]")
        print("\nRequires environment variables:")
        print("  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        print("\nExample:")
        print("  export DB_HOST=db-postgresql-nyc1-12345.b.db.ondigitalocean.com")
        print("  export DB_USER=doadmin")
        print("  export DB_PASSWORD=your_password")
        print("  python production_speaker_detector.py podcast.mp4 transcript.json")
        sys.exit(1)
    
    video_file = sys.argv[1]
    transcript_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "synced_output.json"
    
    # Database configuration from environment
    db_config = {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', '25060')),
        'database': os.getenv('DB_NAME', 'defaultdb'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    # Validate config
    if not all([db_config['host'], db_config['user'], db_config['password']]):
        print("Missing database configuration!")
        print("Set DB_HOST, DB_USER, DB_PASSWORD environment variables")
        sys.exit(1)
    
    print("="*80)
    print("Production Video-Audio Speaker Synchronization")
    print("with InsightFace (ArcFace) and Akamai Database")
    print("="*80)
    
    result = sync_video_with_transcript_production(
        video_path=video_file,
        transcript_path=transcript_file,
        db_config=db_config,
        output_path=output_file,
        device='cuda' if os.getenv('USE_GPU', '').lower() == 'true' else 'cpu'
    )
    
    print("\n" + "="*80)
    print("Synchronization Complete!")
    print("="*80)
    
    summary = result['correction_summary']
    print(f"\nUnique persons in database: {result['total_unique_persons']}")
    print(f"Total utterances: {summary['total_utterances']}")
    print(f"Video-matched: {summary['video_matched']}")
    print(f"Match rate: {summary['match_rate']:.1%}")
    print(f"\nSpeaker distribution:")
    for speaker, count in summary['speaker_distribution'].items():
        print(f"  Person {speaker}: {count} utterances")