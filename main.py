import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import os

# Optional HTTP client for backend integration
try:
    import requests
    _requests_available = True
except Exception:
    requests = None
    _requests_available = False

class FitnessTrainer:
    """Enhanced FitnessTrainer with better camera handling and error recovery."""
    
    def __init__(self, camera_index=0, width=640, height=480, camera_backend=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # TTS (make it optional to avoid blocking)
        try:
            self.engine = pyttsx3.init()
            self.tts_enabled = True
        except:
            self.tts_enabled = False
            print("⚠️  TTS not available")
        
        # Camera setup with better error handling
        self.camera_index = camera_index
        self.camera_backend = camera_backend
        self.width = width
        self.height = height
        self.cap = None
        self.camera_initialized = False
        
        # State variables
        self.correct = 0
        self.incorrect = 0
        self.stage = None
        # Track squat progress for lenient counting
        self.squat_started = False
        self.min_knee_angle = 180.0
        # Timing helpers to reduce false positives
        self._s2_enter_time = None
        self._last_stage = None
        # Leniency / thresholds (defaults can be overridden)
        self.s3_cutoff = 95            # knee angle cutoff to consider S3
        self.min_knee_threshold = 110  # observed min knee angle allowed for counting
        self.back_min = 15
        self.back_max = 55
        # Dwell time (ms) required to be in S3 before allowing completion
        self.s3_dwell_ms = 0
        self._s3_enter_time = None
        # Sequence tracking for strict counting
        self.state_seq = []
        self.last_count_time = 0
        self.count_cooldown_ms = 800
        self.saw_s3_since_last_s1 = False

        # Session logging
        self.log_csv = False
        self.csv_path = 'session_log.csv'
        self.feedback = "Initializing..."
        
        # Threading
        self.running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None

        # Backend integration (optional)
        # Provide BACKEND_URL environment variable like: http://127.0.0.1:5000
        self.backend_url = os.getenv('BACKEND_URL')
        self.session_id = None
        self.backend_enabled = bool(self.backend_url and _requests_available)
        
        # Initialize camera
        self.init_camera()
        
    def init_camera(self):
        """Robust camera initialization: try multiple indices and backends.

        This method will:
        - Release any previous capture object
        - Probe the requested index first, then common indices (0-3)
        - Try DirectShow, MSMF (if available) and the default backend
        - Read a few frames to verify the stream before accepting it
        - Set camera properties on success and provide clear feedback
        """
        # Release any previous capture
        try:
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
        except Exception:
            pass

        self.camera_initialized = False
        self.feedback = "Probing camera..."

        # Build index list (prefer configured index)
        indices_to_try = [self.camera_index]
        for i in range(0, 4):
            if i not in indices_to_try:
                indices_to_try.append(i)

        # Build backend list - PREFER DirectShow on Windows (more stable)
        backends_to_try = []
        if self.camera_backend == 'dshow' and hasattr(cv2, 'CAP_DSHOW'):
            backends_to_try.append(cv2.CAP_DSHOW)
        elif self.camera_backend == 'msmf' and hasattr(cv2, 'CAP_MSMF'):
            backends_to_try.append(cv2.CAP_MSMF)
        else:
            # Default: Try DirectShow FIRST (best for Windows stability)
            if hasattr(cv2, 'CAP_DSHOW'):
                backends_to_try.append(cv2.CAP_DSHOW)
            if hasattr(cv2, 'CAP_MSMF'):
                backends_to_try.append(cv2.CAP_MSMF)
        # Fallback to default backend only if others fail
        backends_to_try.append(None)

        for idx in indices_to_try:
            for backend in backends_to_try:
                cap = None
                try:
                    if backend is not None:
                        cap = cv2.VideoCapture(idx, backend)
                        backend_name = "DirectShow" if backend == getattr(cv2, 'CAP_DSHOW', None) else "MSMF"
                    else:
                        cap = cv2.VideoCapture(idx)
                        backend_name = "Default"

                    if not cap or not cap.isOpened():
                        try:
                            if cap:
                                cap.release()
                        except Exception:
                            pass
                        continue

                    # Try reading a few frames to ensure the stream is alive
                    ok = False
                    for _ in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            ok = True
                            break
                        time.sleep(0.1)

                    if not ok:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        continue

                    # Configure and accept this capture - CRITICAL for stability
                    try:
                        # Set resolution FIRST
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        
                        # CRITICAL: Disable buffering completely (buffer size = 1)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Force 30 FPS for stability
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Try to disable auto-exposure (reduces flicker/corruption)
                        try:
                            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        except:
                            pass
                        
                        # Set FOURCC codec to MJPEG for better compatibility
                        try:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                        except:
                            pass
                        
                        # Disable any conversion
                        try:
                            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                        except:
                            pass
                            
                    except Exception as e:
                        print(f"⚠️  Some camera properties couldn't be set: {e}")

                    self.cap = cap
                    self.camera_index = idx
                    self.camera_initialized = True
                    self.feedback = f"Camera ready (index {idx}, {backend_name})"
                    print(f"✅ Camera initialized at index {idx} with {backend_name} backend")
                    return

                except Exception as e:
                    print(f"❌ Error probing camera index {idx} backend {backend}: {e}")
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
                    continue

        # If we reach here no camera was found
        self.camera_initialized = False
        self.feedback = "Camera not available"
        print("❌ Could not initialize any camera")
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def analyze_squat(self, landmarks):
        """Analyze squat form and provide feedback."""
        try:
            # Get coordinates with visibility check
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            # Check if landmarks are visible (mandatory: > 0.8)
            vis_thresh = 0.8
            if (left_shoulder.visibility < vis_thresh or left_hip.visibility < vis_thresh or 
                left_knee.visibility < vis_thresh or left_ankle.visibility < vis_thresh):
                self.feedback = "Position yourself fully in frame"
                return 180, 0
            
            # Convert to coordinate arrays
            shoulder_coord = [left_shoulder.x, left_shoulder.y]
            hip_coord = [left_hip.x, left_hip.y]
            knee_coord = [left_knee.x, left_knee.y]
            ankle_coord = [left_ankle.x, left_ankle.y]
            
            # Calculate angles
            knee_angle = self.calculate_angle(hip_coord, knee_coord, ankle_coord)
            back_angle = self.calculate_angle(shoulder_coord, hip_coord, knee_coord)
            
            # State machine logic (using configured cutoffs)
            current_stage = None
            # Standing if very open
            if knee_angle > 160:
                current_stage = "S1"
            # Transition zone (use s3_cutoff to determine S2/S3 boundary)
            elif knee_angle > self.s3_cutoff:
                current_stage = "S2"
            else:
                # Treat anything <= s3_cutoff as squat depth (S3)
                current_stage = "S3"

            # Update min_knee_angle when in squat (used for lenient counting)
            if current_stage == "S3":
                # Mark start time when entering S3
                if not self.squat_started:
                    self._s3_enter_time = int(time.time() * 1000)
                # Reset S2 timer when we actually enter S3
                self._s2_enter_time = None
                self.squat_started = True
                if knee_angle < self.min_knee_angle:
                    self.min_knee_angle = knee_angle
            elif current_stage == "S2":
                # record when we moved into S2 (descending)
                if self._last_stage != "S2":
                    self._s2_enter_time = int(time.time() * 1000)
            
            # Feedback logic - follow exact requested messages
            feedback = ""
            if back_angle < 20:
                feedback = "Bend forward"
            elif back_angle > 45:
                feedback = "Bend backwards"
            # Stage-specific cues
            if current_stage == "S2" and knee_angle > 80:
                feedback = "Lower your hips"
            if current_stage == "S3" and knee_angle < 50:
                feedback = "Squat too deep"
            # Knees over toes check (normalized x coords)
            try:
                if left_ankle.x > left_knee.x:
                    feedback = "Knees over toes"
            except Exception:
                pass
            
            # Sequence-based counting (mandatory): maintain state_seq and detect S2->S3->S2
            # Only append when stage changes
            if not self.state_seq or self.state_seq[-1] != current_stage:
                self.state_seq.append(current_stage)
                # cap sequence length
                if len(self.state_seq) > 10:
                    self.state_seq = self.state_seq[-10:]

            # Detect correct rep: last three states = [S2,S3,S2]
            now_ms = int(time.time() * 1000)
            if len(self.state_seq) >= 3 and self.state_seq[-3:] == ["S2", "S3", "S2"]:
                # require that we actually remained in S3 for the configured dwell time
                s3_ok = True
                if self.s3_dwell_ms and self._s3_enter_time:
                    s3_ok = (now_ms - self._s3_enter_time) >= int(self.s3_dwell_ms)

                # cooldown guard and minimum S3 dwell
                if s3_ok and (now_ms - self.last_count_time > self.count_cooldown_ms):
                    self.correct += 1
                    self.last_count_time = now_ms
                    feedback = "Great squat!"
                    # TTS in background
                    if self.tts_enabled:
                        threading.Thread(target=self.speak_feedback, args=(feedback,), daemon=True).start()
                    # Log
                    if self.log_csv:
                        self._log_event('correct')
                    # Send update to backend (non-blocking)
                    try:
                        if self.backend_enabled:
                            threading.Thread(target=self._send_count_update, args=(True,), daemon=True).start()
                    except Exception:
                        pass
                    # reset squat trackers
                    self.squat_started = False
                    self.min_knee_angle = 180.0
                    self._s3_enter_time = None

            # Incorrect squat: returned to S1 without seeing S3 since last descent
            # Only count as incorrect if we actually spent a short time in S2 (to avoid 'scanning' motions)
            if current_stage == "S1":
                recent = self.state_seq[-8:]
                saw_s2 = "S2" in recent
                saw_s3 = "S3" in recent
                # ensure S2 was held for at least 250ms before returning to S1
                s2_ok = False
                if self._s2_enter_time:
                    s2_ok = (now_ms - self._s2_enter_time) >= 250

                # only mark incorrect if we saw S2, did NOT see S3, S2 was held briefly, and cooldown passed
                if saw_s2 and not saw_s3 and s2_ok and (now_ms - self.last_count_time > self.count_cooldown_ms):
                    # also ensure we weren't barely moving (min_knee_angle must have dropped somewhat)
                    if self.min_knee_angle < (self.min_knee_threshold + 10):
                        self.incorrect += 1
                        self.last_count_time = now_ms
                        feedback = "Incomplete squat"
                        if self.tts_enabled:
                            threading.Thread(target=self.speak_feedback, args=(feedback,), daemon=True).start()
                        if self.log_csv:
                            self._log_event('incorrect')
                            # Send update to backend (non-blocking)
                            try:
                                if self.backend_enabled:
                                    threading.Thread(target=self._send_count_update, args=(False,), daemon=True).start()
                            except Exception:
                                pass

            # Reset squat tracking when we are standing or after a completion
            if current_stage == "S1":
                self.squat_started = False
                self.min_knee_angle = 180.0
                self._s3_enter_time = None
                self._s2_enter_time = None

            # track the last stage for transitions
            self._last_stage = current_stage
            
            self.stage = current_stage
            self.feedback = feedback
            
            return knee_angle, back_angle
            
        except Exception as e:
            self.feedback = f"Analysis error: {str(e)}"
            return 180, 0
    
    def speak_feedback(self, text):
        """Speak feedback using TTS (non-blocking)."""
        if self.tts_enabled:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass

    # --- Backend helper methods ---
    def _create_remote_session(self):
        """Create a session on the backend and store the session id."""
        if not self.backend_enabled:
            return
        try:
            url = f"{self.backend_url.rstrip('/')}/sessions"
            payload = {
                'correct': int(self.correct),
                'incorrect': int(self.incorrect),
                'feedback': self.feedback,
            }
            resp = requests.post(url, json=payload, timeout=3)
            if resp.status_code in (200, 201):
                data = resp.json()
                self.session_id = data.get('id')
        except Exception:
            # non-fatal
            pass

    def _update_remote_session(self, ended=False):
        """Send a PATCH request to update counts/ended_at for current session."""
        if not self.backend_enabled or not self.session_id:
            return
        try:
            url = f"{self.backend_url.rstrip('/')}/sessions/{self.session_id}"
            payload = {
                'correct': int(self.correct),
                'incorrect': int(self.incorrect),
            }
            if ended:
                from datetime import datetime
                payload['ended_at'] = datetime.utcnow().isoformat()
            requests.patch(url, json=payload, timeout=3)
        except Exception:
            pass

    def _send_count_update(self, correct_incremented: bool):
        """Non-blocking helper to create session if needed and update counts."""
        try:
            # ensure remote session exists
            if not self.session_id:
                self._create_remote_session()
            # send counts
            self._update_remote_session(ended=False)
        except Exception:
            pass

    def _draw_status_overlay(self, image):
        """Draw a semi-opaque status box with counters and feedback on the frame."""
        try:
            h, w = image.shape[:2]
            # Box dimensions
            box_w = 260
            box_h = 110
            box_x = 10
            box_y = 10

            # Semi-opaque background
            overlay = image.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Text values
            font = cv2.FONT_HERSHEY_SIMPLEX
            y = box_y + 25
            cv2.putText(image, f'Correct: {self.correct}', (box_x + 10, y), font, 0.7, (0, 255, 0), 2)
            y += 28
            cv2.putText(image, f'Incorrect: {self.incorrect}', (box_x + 10, y), font, 0.7, (0, 0, 255), 2)
            y += 28
            cv2.putText(image, f'Stage: {self.stage or "-"}', (box_x + 10, y), font, 0.65, (255, 200, 0), 2)

            # Feedback on separate line at bottom-left
            try:
                feedback_text = str(self.feedback)[:60]
                cv2.putText(image, feedback_text, (10, h - 10), font, 0.6, (0, 255, 255), 2)
            except Exception:
                pass
        except Exception:
            pass

    def _log_event(self, kind):
        """Append a CSV row with timestamp, kind, correct, incorrect."""
        try:
            import csv
            ts = int(time.time() * 1000)
            row = [ts, kind, self.correct, self.incorrect]
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f"❌ Logging error: {e}")
    
    def create_placeholder_frame(self, message="Camera not available"):
        """Create a placeholder frame when camera is not working."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # Calculate text size and position for centering
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = (frame.shape[0] + text_size[1]) // 2
        
        cv2.putText(frame, message, (x, y), font, font_scale, color, thickness)
        
        # Add some visual elements
        cv2.rectangle(frame, (50, 50), (590, 430), (100, 100, 100), 2)
        
        return frame
    
    def process_frame(self, image):
        """Process a single frame for pose detection."""
        if image is None or image.size == 0:
            return self.create_placeholder_frame("No camera input")
        
        # Validate frame dimensions and integrity
        if len(image.shape) != 3 or image.shape[2] != 3:
            return self.create_placeholder_frame("Invalid frame format")
            
        try:
            # CRITICAL: Create a deep copy to avoid any buffer corruption
            image = np.copy(image)
            
            # Enhanced corruption detection
            h, w = image.shape[:2]
            
            # Check frame is reasonable size
            if h < 100 or w < 100:
                return self.create_placeholder_frame("Invalid frame size")
            
            # Check for corruption in bottom region (common corruption area)
            bottom_half = image[h//2:, :]
            bottom_nonzero = np.count_nonzero(bottom_half)
            bottom_total = bottom_half.size
            
            if bottom_nonzero < (bottom_total * 0.15):
                return self.create_placeholder_frame("Corrupted frame (bottom region)")
            
            # Check overall frame validity
            if np.count_nonzero(image) < (image.size * 0.1):
                return self.create_placeholder_frame("Corrupted frame detected")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Make detection
            results = self.pose.process(image_rgb)
            
            # Draw the pose annotation on the image
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Analyze squat
                landmarks = results.pose_landmarks.landmark
                knee_angle, back_angle = self.analyze_squat(landmarks)
                
                # Draw status overlay (counters/feedback) - single clean overlay box
                try:
                    self._draw_status_overlay(image_bgr)
                except Exception:
                    pass
            else:
                # No pose detected
                cv2.putText(image_bgr, "No person detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(image_bgr, "Stand in front of camera", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image_bgr
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return self.create_placeholder_frame(f"Processing error: {str(e)}")
    
    def run_processing(self):
        """Main processing loop running in background thread."""
        consecutive_failures = 0
        max_failures = 30  # Allow 30 consecutive failures before giving up
        frame_skip = 0  # Skip corrupted frames
        
        while self.running:
            if not self.camera_initialized or not self.cap or not self.cap.isOpened():
                # Try to reinitialize camera
                self.init_camera()
                if not self.camera_initialized:
                    # Create placeholder frame
                    placeholder = self.create_placeholder_frame("Reconnecting camera...")
                    with self.frame_lock:
                        self.current_frame = placeholder.copy()
                    time.sleep(1)
                    continue
            
            try:
                # Aggressive buffer clearing - flush old frames completely
                for _ in range(4):  # Increased from 2 to 4
                    self.cap.grab()
                
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None and frame.size > 0:
                    # Enhanced frame validation
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        h, w = frame.shape[:2]
                        
                        # Check if frame dimensions are correct
                        if h < 100 or w < 100:
                            frame_skip += 1
                            continue
                        
                        # Check for corruption in different regions of the frame
                        # Split frame into thirds vertically
                        third_h = h // 3
                        top_third = frame[0:third_h, :]
                        middle_third = frame[third_h:2*third_h, :]
                        bottom_third = frame[2*third_h:, :]
                        
                        # Check each region for corruption (too many zeros = noise/corruption)
                        top_valid = np.count_nonzero(top_third) > (top_third.size * 0.15)
                        middle_valid = np.count_nonzero(middle_third) > (middle_third.size * 0.15)
                        bottom_valid = np.count_nonzero(bottom_third) > (bottom_third.size * 0.15)
                        
                        # Frame is only valid if all regions pass
                        if top_valid and middle_valid and bottom_valid:
                            consecutive_failures = 0
                            frame_skip = 0
                            
                            # Extra safety: make a deep copy immediately
                            frame_copy = np.copy(frame)
                            processed_frame = self.process_frame(frame_copy)
                            
                            with self.frame_lock:
                                self.current_frame = np.copy(processed_frame)
                        else:
                            frame_skip += 1
                            if frame_skip > 5:
                                print(f"⚠️  Corrupted frames detected (top:{top_valid}, mid:{middle_valid}, bot:{bottom_valid})")
                                # Force camera reinit on persistent corruption
                                if frame_skip > 15:
                                    print(f"🔄 Reinitializing camera due to persistent corruption")
                                    self.camera_initialized = False
                                    frame_skip = 0
                    else:
                        frame_skip += 1
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        self.camera_initialized = False
                        self.feedback = "Camera connection lost"
                        print(f"❌ Camera connection lost after {max_failures} failures")
                        
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ Frame capture error: {e}")
                
                if consecutive_failures > max_failures:
                    self.camera_initialized = False
                    self.feedback = f"Camera error: {str(e)}"
            
            time.sleep(0.033)  # ~30 FPS processing (more stable)
    
    def start(self):
        """Start background processing."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_processing, daemon=True)
            self.thread.start()
            # Create remote session record (non-blocking)
            try:
                if self.backend_enabled:
                    threading.Thread(target=self._create_remote_session, daemon=True).start()
            except Exception:
                pass
            print("▶️  Training started")
    
    def stop(self):
        """Stop background processing."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        # Finalize remote session (non-blocking)
        try:
            if self.backend_enabled:
                threading.Thread(target=self._update_remote_session, kwargs={'ended': True}, daemon=True).start()
        except Exception:
            pass
        print("⏹️  Training stopped")
    
    def get_frame(self):
        """Get latest frame as JPEG bytes for streaming."""
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    # Use higher quality encoding and proper parameters
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90,
                                   int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
                    success, buffer = cv2.imencode('.jpg', self.current_frame, encode_param)
                    
                    if success and buffer is not None:
                        return buffer.tobytes()
                    else:
                        print(f"⚠️  Frame encoding failed")
                        # Return placeholder frame
                        placeholder = self.create_placeholder_frame("Encoding error")
                        _, buffer = cv2.imencode('.jpg', placeholder, encode_param)
                        return buffer.tobytes()
                except Exception as e:
                    print(f"❌ Frame encoding error: {e}")
                    # Return placeholder frame
                    placeholder = self.create_placeholder_frame("Encoding error")
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    return buffer.tobytes()
        return None
    
    def get_status(self):
        """Get current status for API."""
        return {
            'correct': self.correct,
            'incorrect': self.incorrect,
            'stage': self.stage or '-',
            'feedback': self.feedback or '-',
            'camera_ready': self.camera_initialized
        }
    
    def release(self):
        """Release resources."""
        self.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print("🧹 Resources released")

def main():
    """Main function for standalone execution.

    Supports command-line args to override camera index/backend and frame size.
    """
    import argparse

    parser = argparse.ArgumentParser(description='AI Fitness Trainer - standalone')
    parser.add_argument('--index', type=int, default=0, help='Camera index to use')
    parser.add_argument('--backend', type=str, default=None, choices=['dshow', 'msmf'], help='Optional camera backend to prefer')
    parser.add_argument('--width', type=int, default=640, help='Requested frame width')
    parser.add_argument('--height', type=int, default=480, help='Requested frame height')
    parser.add_argument('--fullscreen', action='store_true', help='Open OpenCV window in fullscreen')
    parser.add_argument('--leniency', type=str, choices=['A', 'B', 'custom'], default=None, help='Leniency preset: A (moderate) or B (very lenient) or custom')
    parser.add_argument('--dwell-ms', type=int, default=0, help='Milliseconds required to hold S3 before counting')
    # Custom numeric overrides (only used if --leniency custom)
    parser.add_argument('--s3-cutoff', type=int, default=None, help='Knee angle cutoff for S3')
    parser.add_argument('--min-knee', type=int, default=None, help='Min knee angle threshold for counting')
    parser.add_argument('--back-min', type=int, default=None, help='Minimum acceptable back angle')
    parser.add_argument('--back-max', type=int, default=None, help='Maximum acceptable back angle')
    
    args = parser.parse_args()

    trainer = FitnessTrainer(camera_index=args.index, width=args.width, height=args.height, camera_backend=args.backend)
    # Apply leniency presets or custom thresholds
    if args.leniency == 'A':
        trainer.s3_cutoff = 110
        trainer.min_knee_threshold = 120
        trainer.back_min = 12
        trainer.back_max = 60
    elif args.leniency == 'B':
        trainer.s3_cutoff = 120
        trainer.min_knee_threshold = 130
        trainer.back_min = 10
        trainer.back_max = 65
    elif args.leniency == 'custom':
        if args.s3_cutoff is not None:
            trainer.s3_cutoff = args.s3_cutoff
        if args.min_knee is not None:
            trainer.min_knee_threshold = args.min_knee
        if args.back_min is not None:
            trainer.back_min = args.back_min
        if args.back_max is not None:
            trainer.back_max = args.back_max

    # Dwell time
    trainer.s3_dwell_ms = max(0, int(args.dwell_ms))
    trainer.start()
    
    try:
        print("🎥 AI Fitness Trainer running. Press 'q' to quit.")
        # Optionally create a fullscreen window
        if args.fullscreen:
            try:
                cv2.namedWindow('AI Fitness Trainer', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('AI Fitness Trainer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass

        while True:
            frame = trainer.get_frame()
            if frame:
                # Convert bytes back to image for display
                nparr = np.frombuffer(frame, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow('AI Fitness Trainer', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        trainer.release()
        cv2.destroyAllWindows()

class JumpTrainer:
    """Vertical Jump Height Measurement Trainer."""
    
    def __init__(self, camera_index=0, width=640, height=480, camera_backend=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # lighter model for real-time
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.camera_index = camera_index
        self.camera_backend = camera_backend
        self.width = width
        self.height = height
        self.cap = None
        self.camera_initialized = False
        
        # Jump tracking state
        self.state = "idle"  # idle -> armed -> airborne -> landed
        self.baseline_nose_y = None
        self.peak_delta_pixels = 0.0
        self.last_jump_inches = 0.0
        self.last_jump_valid = False
        self.last_jump_reason = ""
        self.airborne_frames = 0
        self.nose_y_buffer = []
        self.prev_dy = None
        
        # Configuration
        self.calibration_inches = 12.0
        self.calibration_pixels = 100.0
        self.min_jump_inches = 2.0
        self.min_airborne_frames = 6
        
        # Threading
        self.running = False
        self.thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.feedback = "Touch right hand to nose to ARM"
        
        # Initialize camera (same as FitnessTrainer)
        self._init_camera()
    
    def _init_camera(self):
        """Initialize camera with retry logic."""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                # Release any existing capture
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    time.sleep(0.2)
                
                if self.camera_backend == 'dshow':
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                elif self.camera_backend == 'msmf':
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)
                
                if self.cap and self.cap.isOpened():
                    # Set resolution first
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    
                    # CRITICAL: Disable buffering
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Force 30 FPS
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Set MJPEG codec for compatibility
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    except:
                        pass
                    
                    # Disable auto exposure
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    except:
                        pass
                    
                    # Disable RGB conversion
                    try:
                        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    except:
                        pass
                    
                    # Test read multiple frames
                    ok = False
                    for _ in range(5):
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            ok = True
                            break
                        time.sleep(0.1)
                    
                    if ok:
                        self.camera_initialized = True
                        print(f"✅ Jump camera initialized at index {self.camera_index} (attempt {attempt + 1})")
                        return
                
                print(f"⚠️  Camera attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)
                
            except Exception as e:
                print(f"❌ Camera initialization error (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay)
        
        print(f"❌ Failed to initialize camera after {max_retries} attempts")
        self.camera_initialized = False
    
    def inches_per_pixel(self):
        return self.calibration_inches / self.calibration_pixels if self.calibration_pixels > 0 else 0.12
    
    def landmark_xy(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            return None
        return (lm.x * w, lm.y * h)
    
    def distance(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def is_hand_to_nose(self, landmarks, w, h, hand_idx, nose_idx=0, thresh=60.0):
        p_hand = self.landmark_xy(landmarks, hand_idx, w, h)
        p_nose = self.landmark_xy(landmarks, nose_idx, w, h)
        if not p_hand or not p_nose:
            return False
        return self.distance(p_hand, p_nose) < thresh
    
    def detect_squat_cheat(self, landmarks, w, h):
        """Detect if landing in a squat position (cheating)."""
        idx_map = {
            'LEFT_HIP': 23, 'RIGHT_HIP': 24,
            'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
        }
        pts = {}
        for name, val in idx_map.items():
            p = self.landmark_xy(landmarks, val, w, h)
            if not p:
                return False
            pts[name] = p
        
        hip_y = (pts["LEFT_HIP"][1] + pts["RIGHT_HIP"][1]) / 2
        knee_y = (pts["LEFT_KNEE"][1] + pts["RIGHT_KNEE"][1]) / 2
        
        # Calculate knee angle
        def angle(a, b, c):
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        
        knee_angle_left = angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"])
        knee_angle_right = angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"])
        knee_angle = (knee_angle_left + knee_angle_right) / 2
        
        is_knee_bent = knee_angle < 150
        is_hip_low = hip_y > (knee_y - 20)
        
        return is_knee_bent and is_hip_low
    
    def process_frame(self, frame):
        """Process a single frame for jump detection."""
        if frame is None or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Validate frame integrity
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # CRITICAL: Deep copy to avoid buffer corruption
        frame = np.copy(frame)
        
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        image_bgr = frame.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Draw pose
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            nose_idx = 0  # NOSE
            right_hand_idx = 20  # RIGHT_INDEX
            left_hand_idx = 19  # LEFT_INDEX
            
            # Control gestures
            if self.state == "idle" and self.is_hand_to_nose(landmarks, w, h, right_hand_idx, nose_idx):
                self.state = "armed"
                p_nose = self.landmark_xy(landmarks, nose_idx, w, h)
                if p_nose:
                    self.baseline_nose_y = p_nose[1]
                    self.peak_delta_pixels = 0.0
                    self.nose_y_buffer = []
                    self.prev_dy = None
                    self.airborne_frames = 0
                self.feedback = "ARMED - Jump when ready!"
            
            if self.state in ("armed", "landed") and self.is_hand_to_nose(landmarks, w, h, left_hand_idx, nose_idx):
                # Reset
                self.state = "idle"
                self.baseline_nose_y = None
                self.peak_delta_pixels = 0.0
                self.last_jump_inches = 0.0
                self.feedback = "Touch right hand to nose to ARM"
            
            # Measurement logic
            p_nose = self.landmark_xy(landmarks, nose_idx, w, h)
            if p_nose and self.baseline_nose_y is not None:
                # Smooth nose y
                self.nose_y_buffer.append(p_nose[1])
                if len(self.nose_y_buffer) > 5:
                    self.nose_y_buffer.pop(0)
                smooth_nose_y = sum(self.nose_y_buffer) / len(self.nose_y_buffer)
                
                dy = self.baseline_nose_y - smooth_nose_y  # positive when moving up
                self.peak_delta_pixels = max(self.peak_delta_pixels, dy)
                
                # State transitions
                arm_thresh = max(8, 0.015 * h)
                land_thresh = max(4, 0.008 * h)
                
                if self.state == "armed" and dy > arm_thresh:
                    self.state = "airborne"
                    self.airborne_frames = 0
                    self.feedback = "AIRBORNE!"
                
                elif self.state == "airborne":
                    self.airborne_frames += 1
                    if dy < land_thresh and self.airborne_frames >= self.min_airborne_frames:
                        # Landing detected
                        self.state = "landed"
                        inches = max(0.0, self.peak_delta_pixels) * self.inches_per_pixel()
                        self.last_jump_inches = inches
                        
                        if inches < self.min_jump_inches:
                            self.last_jump_valid = False
                            self.last_jump_reason = "too-small"
                            self.feedback = f"Jump too small: {inches:.1f} in"
                        else:
                            cheated = self.detect_squat_cheat(landmarks, w, h)
                            if cheated:
                                self.last_jump_valid = False
                                self.last_jump_reason = "squat-cheat"
                                self.feedback = f"INVALID (squat): {inches:.1f} in"
                            else:
                                self.last_jump_valid = True
                                self.last_jump_reason = "ok"
                                self.feedback = f"VALID JUMP: {inches:.1f} inches!"
        else:
            self.feedback = "No person detected"
        
        # Draw overlay
        self._draw_overlay(image_bgr)
        
        return image_bgr
    
    def _draw_overlay(self, image):
        """Draw status overlay."""
        h, w = image.shape[:2]
        
        # Semi-transparent box
        box_w = 300
        box_h = 140
        box_x = 10
        box_y = 10
        
        overlay = image.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = box_y + 25
        
        # State
        color = (0, 255, 255) if self.state == "armed" else (255, 200, 0) if self.state == "airborne" else (200, 200, 200)
        cv2.putText(image, f'State: {self.state.upper()}', (box_x + 10, y), font, 0.7, color, 2)
        y += 30
        
        # Last jump result
        if self.last_jump_inches > 0:
            jump_color = (0, 255, 0) if self.last_jump_valid else (0, 0, 255)
            status_text = "VALID" if self.last_jump_valid else "INVALID"
            cv2.putText(image, f'Last: {status_text}', (box_x + 10, y), font, 0.6, jump_color, 2)
            y += 28
            cv2.putText(image, f'Height: {self.last_jump_inches:.1f} in', (box_x + 10, y), font, 0.6, jump_color, 2)
        
        # Feedback at bottom
        cv2.putText(image, self.feedback, (10, h - 10), font, 0.6, (0, 255, 255), 2)
        
        # Instructions at bottom
        cv2.putText(image, "Right hand->nose: ARM", (10, h - 70), font, 0.5, (200, 200, 200), 1)
        cv2.putText(image, "Left hand->nose: RESET", (10, h - 50), font, 0.5, (200, 200, 200), 1)
    
    def run_processing(self):
        """Main processing loop."""
        frame_skip = 0
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                
                # Aggressive buffer clearing (4 frames)
                for _ in range(4):
                    self.cap.grab()
                
                ret, frame = self.cap.retrieve()
                
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.03)
                    continue
                
                # Enhanced frame validation
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    h, w = frame.shape[:2]
                    
                    # Check dimensions
                    if h < 100 or w < 100:
                        frame_skip += 1
                        continue
                    
                    # Check for regional corruption
                    third_h = h // 3
                    top_third = frame[0:third_h, :]
                    middle_third = frame[third_h:2*third_h, :]
                    bottom_third = frame[2*third_h:, :]
                    
                    top_valid = np.count_nonzero(top_third) > (top_third.size * 0.15)
                    middle_valid = np.count_nonzero(middle_third) > (middle_third.size * 0.15)
                    bottom_valid = np.count_nonzero(bottom_third) > (bottom_third.size * 0.15)
                    
                    if top_valid and middle_valid and bottom_valid:
                        frame_skip = 0
                        # Deep copy for safety
                        frame_copy = np.copy(frame)
                        processed_frame = self.process_frame(frame_copy)
                        
                        with self.frame_lock:
                            self.current_frame = np.copy(processed_frame)
                    else:
                        frame_skip += 1
                        if frame_skip > 15:
                            print(f"⚠️  Jump: Persistent frame corruption, reinitializing")
                            self._init_camera()
                            frame_skip = 0
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start background processing."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_processing, daemon=True)
            self.thread.start()
            print("▶️  Jump training started")
    
    def stop(self):
        """Stop background processing."""
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
            print("⏹️  Jump training stopped")
    
    def get_frame(self):
        """Get latest frame as JPEG bytes."""
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    # Use higher quality encoding and proper parameters
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90,
                                   int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
                    success, buffer = cv2.imencode('.jpg', self.current_frame, encode_param)
                    
                    if success and buffer is not None:
                        return buffer.tobytes()
                    else:
                        print(f"⚠️  Frame encoding failed")
                        return None
                except Exception as e:
                    print(f"❌ Frame encoding error: {e}")
                    return None
        return None
    
    def get_status(self):
        """Get current status for API."""
        return {
            'state': self.state,
            'last_jump_inches': round(self.last_jump_inches, 2),
            'last_jump_valid': self.last_jump_valid,
            'last_jump_reason': self.last_jump_reason,
            'feedback': self.feedback,
            'camera_ready': self.camera_initialized
        }
    
    def release(self):
        """Release resources."""
        self.stop()
        try:
            if hasattr(self, 'pose') and self.pose:
                self.pose.close()
        except Exception as e:
            print(f"⚠️  Error closing pose: {e}")
        
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
        except Exception as e:
            print(f"⚠️  Error releasing camera: {e}")
        
        print("🧹 Jump trainer resources released")


if __name__ == "__main__":
    main()