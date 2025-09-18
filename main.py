import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

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

        # Build backend list
        backends_to_try = []
        if self.camera_backend == 'dshow' and hasattr(cv2, 'CAP_DSHOW'):
            backends_to_try.append(cv2.CAP_DSHOW)
        if self.camera_backend == 'msmf' and hasattr(cv2, 'CAP_MSMF'):
            backends_to_try.append(cv2.CAP_MSMF)
        if hasattr(cv2, 'CAP_DSHOW') and cv2.CAP_DSHOW not in backends_to_try:
            backends_to_try.append(cv2.CAP_DSHOW)
        if hasattr(cv2, 'CAP_MSMF') and cv2.CAP_MSMF not in backends_to_try:
            backends_to_try.append(cv2.CAP_MSMF)
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
                    for _ in range(3):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            ok = True
                            break
                        time.sleep(0.1)

                    if not ok:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        continue

                    # Configure and accept this capture
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        # Not all backends support buffersize; ignore errors
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                    except Exception:
                        pass

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
        if image is None:
            return self.create_placeholder_frame("No camera input")
            
        try:
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
                
                # Add text overlays with better positioning
                overlay_y = 30
                line_height = 30
                
                cv2.putText(image_bgr, f'Correct: {self.correct}', (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                overlay_y += line_height
                
                cv2.putText(image_bgr, f'Incorrect: {self.incorrect}', (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                overlay_y += line_height
                
                cv2.putText(image_bgr, f'Stage: {self.stage or "-"}', (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                overlay_y += line_height
                
                cv2.putText(image_bgr, f'Knee: {int(knee_angle)}°', (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                overlay_y += 20
                
                cv2.putText(image_bgr, f'Back: {int(back_angle)}°', (10, overlay_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Ensure status overlay (counters/feedback) is always drawn and visible
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
            return self.create_placeholder_frame(f"Processing error: {str(e)}")
    
    def run_processing(self):
        """Main processing loop running in background thread."""
        consecutive_failures = 0
        max_failures = 30  # Allow 30 consecutive failures before giving up
        
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
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0
                    processed_frame = self.process_frame(frame)
                    
                    with self.frame_lock:
                        self.current_frame = processed_frame.copy()
                else:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        self.camera_initialized = False
                        self.feedback = "Camera connection lost"
                        
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ Frame processing error: {e}")
                
                if consecutive_failures > max_failures:
                    self.camera_initialized = False
                    self.feedback = f"Camera error: {str(e)}"
            
            time.sleep(0.01)  # ~100 FPS processing
    
    def start(self):
        """Start background processing."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run_processing, daemon=True)
            self.thread.start()
            print("▶️  Training started")
    
    def stop(self):
        """Stop background processing."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        print("⏹️  Training stopped")
    
    def get_frame(self):
        """Get latest frame as JPEG bytes for streaming."""
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', self.current_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
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

if __name__ == "__main__":
    main()