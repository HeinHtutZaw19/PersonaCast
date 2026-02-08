"""
Video Speaker Detection with Production-Grade Face Recognition

Features:
- InsightFace (ArcFace) for 512-dimensional face embeddings
- Akamai (Linode) Database Cluster (PostgreSQL) for persistent storage
- Person re-identification across videos and sessions
- Multi-shot support with consistent person tracking
- Lip movement detection and speaker synchronization

Requirements:
- insightface
- onnxruntime or onnxruntime-gpu
- psycopg2-binary (PostgreSQL client)
- models/buffalo_l (InsightFace model)
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime
import hashlib
import psycopg2
from psycopg2.extras import execute_values, Json
from contextlib import contextmanager

import insightface
from insightface.app import FaceAnalysis

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class AkamaiDatabaseManager:
    """
    Manages face embeddings in Akamai (Linode) Database Cluster
    PostgreSQL-based persistent storage for person re-identification
    """
    
    def __init__(self, db_config):
        """
        Initialize database connection
        
        Args:
            db_config: dict with keys:
                - host: Database host (e.g., 'db-postgresql-nyc1-12345.b.db.ondigitalocean.com')
                - port: Database port (default: 25060 for Akamai/Linode)
                - database: Database name
                - user: Database user
                - password: Database password
                - sslmode: SSL mode (default: 'require')
        """
        self.db_config = {
            'host': db_config.get('host'),
            'port': db_config.get('port', 25060),
            'database': db_config.get('database'),
            'user': db_config.get('user'),
            'password': db_config.get('password'),
            'sslmode': db_config.get('sslmode', 'require'),
            'connect_timeout': 10
        }
        
        # Test connection and create tables
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.db_config)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    def _initialize_database(self):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Ensure pgvector exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS persons (
                        person_id SERIAL PRIMARY KEY,
                        person_uuid VARCHAR(64) UNIQUE NOT NULL,
                        name VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)

                # IMPORTANT: embedding_vector is vector(512)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        embedding_id SERIAL PRIMARY KEY,
                        person_id INTEGER REFERENCES persons(person_id) ON DELETE CASCADE,
                        embedding_vector vector(512),
                        video_id VARCHAR(255),
                        frame_number INTEGER,
                        timestamp DOUBLE PRECISION,
                        quality_score DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_person_id
                    ON face_embeddings(person_id)
                """)

                # Vector index (cosine)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS face_embeddings_vec_idx
                    ON face_embeddings
                    USING ivfflat (embedding_vector vector_cosine_ops)
                    WITH (lists = 100)
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS videos (
                        video_id SERIAL PRIMARY KEY,
                        video_uuid VARCHAR(64) UNIQUE NOT NULL,
                        video_path VARCHAR(1024),
                        duration DOUBLE PRECISION,
                        fps DOUBLE PRECISION,
                        total_persons INTEGER,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS utterances (
                        utterance_id SERIAL PRIMARY KEY,
                        video_uuid VARCHAR(64),
                        person_id INTEGER REFERENCES persons(person_id),
                        start_time DOUBLE PRECISION,
                        end_time DOUBLE PRECISION,
                        transcript TEXT,
                        original_speaker INTEGER,
                        confidence DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                print("✓ Database tables initialized (pgvector)")

    def create_person(self, person_uuid=None, name=None, metadata=None):
        """
        Create a new person in database
        
        Args:
            person_uuid: Optional UUID for person
            name: Optional name
            metadata: Optional metadata dict
            
        Returns:
            int: person_id
        """
        if person_uuid is None:
            person_uuid = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO persons (person_uuid, name, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING person_id
                """, (person_uuid, name, Json(metadata or {})))
                
                person_id = cur.fetchone()[0]
                print(f"✓ Created person {person_id} (UUID: {person_uuid})")
                return person_id
    
    def add_face_embedding(self, person_id, embedding, video_id=None,
                       frame_number=None, timestamp=None, quality_score=None,
                       metadata=None):
        emb = np.asarray(embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        emb_list = emb.tolist()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings
                    (person_id, embedding_vector, video_id, frame_number, timestamp, quality_score, metadata)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                """, (
                    person_id,
                    emb_list,
                    video_id,
                    frame_number,
                    timestamp,
                    quality_score,
                    Json(metadata or {})
                ))

    def find_similar_person(self, embedding, threshold=0.6, top_k=5, probes=10):
        """
        Returns list of (person_id, similarity) sorted by best match.

        threshold is cosine similarity (0..1). With ArcFace, 0.35-0.6 is a typical range.
        """
        emb = np.asarray(embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)  # IMPORTANT: normalize
        emb_list = emb.tolist()

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Improve recall for ivfflat
                cur.execute("SET ivfflat.probes = %s;", (probes,))

                # Find nearest embeddings, then collapse to best per person
                cur.execute(
                    """
                    SELECT person_id, MAX(similarity) AS similarity
                    FROM (
                        SELECT
                            person_id,
                            1 - (embedding_vector <=> %s::vector) AS similarity
                        FROM face_embeddings
                        ORDER BY embedding_vector <=> %s::vector
                        LIMIT %s
                    ) t
                    GROUP BY person_id
                    HAVING MAX(similarity) >= %s
                    ORDER BY similarity DESC
                    LIMIT %s;
                    """,
                    (emb_list, emb_list, top_k * 20, threshold, top_k),
                )

                rows = cur.fetchall()
                return [(int(pid), float(sim)) for pid, sim in rows]

    def get_person_embeddings(self, person_id):
        """Get all embeddings for a person"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT embedding_vector, video_id, frame_number, 
                           timestamp, quality_score
                    FROM face_embeddings
                    WHERE person_id = %s
                    ORDER BY created_at DESC
                """, (person_id,))
                
                results = cur.fetchall()
                return [{
                    'embedding': np.array(row[0]),
                    'video_id': row[1],
                    'frame_number': row[2],
                    'timestamp': row[3],
                    'quality_score': row[4]
                } for row in results]
    
    def save_video_metadata(self, video_uuid, video_path, duration, fps, 
                           total_persons, metadata=None):
        """Save video processing metadata"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO videos 
                    (video_uuid, video_path, duration, fps, total_persons, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_uuid) DO UPDATE SET
                        processed_at = CURRENT_TIMESTAMP,
                        total_persons = EXCLUDED.total_persons,
                        metadata = EXCLUDED.metadata
                """, (video_uuid, video_path, duration, fps, 
                      total_persons, Json(metadata or {})))
    
    def save_utterances(self, video_uuid, utterances):
        """Batch save corrected utterances"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO utterances 
                    (video_uuid, person_id, start_time, end_time, 
                     transcript, original_speaker, confidence)
                    VALUES %s
                """, [(
                    video_uuid,
                    utt.get('speaker'),
                    utt['start'],
                    utt['end'],
                    utt['transcript'],
                    utt.get('original_audio_speaker'),
                    utt.get('speaker_match_confidence', 0.0)
                ) for utt in utterances])
    
    def get_person_statistics(self):
        """Get statistics about all persons in database"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        p.person_id,
                        p.person_uuid,
                        p.name,
                        COUNT(DISTINCT fe.video_id) as video_count,
                        COUNT(fe.embedding_id) as embedding_count,
                        COUNT(DISTINCT u.utterance_id) as utterance_count,
                        p.created_at
                    FROM persons p
                    LEFT JOIN face_embeddings fe ON p.person_id = fe.person_id
                    LEFT JOIN utterances u ON p.person_id = u.person_id
                    GROUP BY p.person_id, p.person_uuid, p.name, p.created_at
                    ORDER BY embedding_count DESC
                """)
                
                results = cur.fetchall()
                return [{
                    'person_id': row[0],
                    'person_uuid': row[1],
                    'name': row[2],
                    'video_count': row[3],
                    'embedding_count': row[4],
                    'utterance_count': row[5],
                    'created_at': row[6].isoformat() if row[6] else None
                } for row in results]


class ProductionFaceRecognitionTracker:
    """
    Production-grade face recognition using InsightFace (ArcFace)
    with Akamai database persistence
    """
    
    def __init__(self, db_manager, model_name='buffalo_l', device='cpu'):
        """
        Initialize InsightFace model and database
        
        Args:
            db_manager: AkamaiDatabaseManager instance
            model_name: InsightFace model name (buffalo_l, buffalo_s, etc.)
            device: 'cpu' or 'cuda'
        """
        self.db_manager = db_manager
        
        # Initialize InsightFace
        print(f"Loading InsightFace model: {model_name}...")
        self.app = FaceAnalysis(name=model_name, providers=[
            'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'
        ])
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        print("✓ InsightFace model loaded")
        
        # Cache for current session (video processing)
        self.session_person_map = {}  # temp_id -> db_person_id
        self.next_temp_id = 0
        
        # MediaPipe for lip detection
        self._init_mediapipe()
    
    def _init_mediapipe(self, landmarker_model_path="models/face_landmarker.task"):
        """Initialize MediaPipe for lip movement detection"""
        if not Path(landmarker_model_path).exists():
            print(f"⚠ MediaPipe model not found: {landmarker_model_path}")
            print("  Lip movement detection will be disabled")
            self.face_landmarker = None
            return
        
        base_options = python.BaseOptions(model_asset_path=str(landmarker_model_path))
        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            running_mode=vision.RunningMode.VIDEO
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)
        
        # Lip landmark indices
        self.upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    
    def extract_face_embedding(self, frame, bbox=None):
        """
        Extract face embedding using InsightFace (ArcFace)
        
        Args:
            frame: Image frame (BGR)
            bbox: Optional bounding box [x, y, w, h]
            
        Returns:
            dict: {
                'embedding': 512-dim numpy array,
                'bbox': [x, y, w, h],
                'quality': float (0-1),
                'landmarks': 5 keypoints
            }
        """
        # Convert to RGB for InsightFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(rgb_frame)
        
        if not faces:
            return None
        
        # If bbox provided, find closest face
        if bbox is not None:
            x, y, w, h = bbox
            cx, cy = x + w/2, y + h/2
            
            min_dist = float('inf')
            best_face = None
            
            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                dist = ((cx - fcx)**2 + (cy - fcy)**2) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    best_face = face
            
            face = best_face
        else:
            # Use first detected face
            face = faces[0]
        
        # Extract embedding (512-dimensional ArcFace feature)
        embedding = face.embedding  # numpy array, shape (512,)
        
        # Get quality score
        quality = float(face.det_score) if hasattr(face, 'det_score') else 0.9
        
        # Get bounding box
        bbox = [int(x) for x in face.bbox]  # [x1, y1, x2, y2]
        
        return {
            'embedding': embedding,
            'bbox': bbox,
            'quality': quality,
            'landmarks': face.kps if hasattr(face, 'kps') else None
        }
    
    def identify_or_register_person(self, embedding, video_id, frame_number, 
                                    timestamp, quality_score, similarity_threshold=0.6):
        """
        Identify person from database or register as new
        
        Args:
            embedding: 512-dim face embedding
            video_id: Current video identifier
            frame_number: Frame number
            timestamp: Timestamp in video
            quality_score: Face quality score
            similarity_threshold: Minimum similarity for match
            
        Returns:
            int: person_id from database
        """
        # Search in database
        matches = self.db_manager.find_similar_person(embedding, threshold=similarity_threshold)
        
        if matches:
            # Found existing person
            person_id, similarity = matches[0]
            print(f"  ✓ Matched person {person_id} (similarity: {similarity:.3f})")
            
            # Add this embedding to their collection
            self.db_manager.add_face_embedding(
                person_id=person_id,
                embedding=embedding,
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                quality_score=quality_score
            )
            
            return person_id
        else:
            # New person - create in database
            person_uuid = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
            person_id = self.db_manager.create_person(
                person_uuid=person_uuid,
                metadata={'first_seen_video': video_id}
            )
            
            # Add first embedding
            self.db_manager.add_face_embedding(
                person_id=person_id,
                embedding=embedding,
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                quality_score=quality_score
            )
            
            return person_id
    
    def calculate_lip_distance(self, landmarks):
        """Calculate lip distance from MediaPipe landmarks"""
        if landmarks is None:
            return 0.0
        
        upper = [landmarks[i] for i in self.upper_lip_indices]
        lower = [landmarks[i] for i in self.lower_lip_indices]
        upper_avg_y = float(np.mean([p.y for p in upper]))
        lower_avg_y = float(np.mean([p.y for p in lower]))
        return abs(lower_avg_y - upper_avg_y)


# Continue in next file due to length...