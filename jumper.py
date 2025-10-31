import os
import cv2
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional configuration from .env (with sensible defaults)
CALIBRATION_INCHES = float(os.getenv("CALIBRATION_INCHES", "12"))  # physical distance represented by CALIBRATION_PIXELS
CALIBRATION_PIXELS = float(os.getenv("CALIBRATION_PIXELS", "100"))  # pixels corresponding to CALIBRATION_INCHES
MIN_JUMP_INCHES = float(os.getenv("MIN_JUMP_INCHES", "2.0"))  # ignore micro movements
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))  # moving average window for vertical signal
# Additional thresholds/tunables
MIN_AIRBORNE_FRAMES = int(os.getenv("MIN_AIRBORNE_FRAMES", "6"))  # frames required in air before landing is valid
VEL_UP_PCT_OF_H = float(os.getenv("VEL_UP_PCT_OF_H", "0.25"))  # upward velocity threshold as fraction of frame height per second
VEL_DOWN_PCT_OF_H = float(os.getenv("VEL_DOWN_PCT_OF_H", "0.25"))  # downward velocity threshold as fraction of frame height per second

# MediaPipe imports (lazy to allow error message if not installed)
try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe is not installed. Please install requirements: pip install -r requirements.txt")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


@dataclass
class JumpResult:
    inches: float
    valid: bool
    reason: str  # "ok", "cheat", "too-small", etc.


def inches_per_pixel() -> float:
    # Simple linear calibration: X pixels == Y inches
    return CALIBRATION_INCHES / CALIBRATION_PIXELS if CALIBRATION_PIXELS > 0 else 0.12


def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(arr) < k:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[k:] - cumsum[:-k]) / float(k)


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_xy(landmarks, idx, w, h) -> Optional[Tuple[float, float]]:
    lm = landmarks[idx]
    # Allow slightly lower visibility to avoid frequent dropouts
    if lm.visibility < 0.4:
        return None
    return (lm.x * w, lm.y * h)


def is_hand_to_nose(landmarks, w, h, hand_idx: int, nose_idx: int = mp_pose.PoseLandmark.NOSE.value, thresh: float = 60.0) -> bool:
    p_hand = landmark_xy(landmarks, hand_idx, w, h)
    p_nose = landmark_xy(landmarks, nose_idx, w, h)
    if not p_hand or not p_nose:
        return False
    return distance(p_hand, p_nose) < thresh


def detect_squat_cheat(landmarks, w, h) -> bool:
    """Detect squatted landing by checking knee-hip-ankle geometry.
    Heuristic:
    - Knee much closer to hip vertically (bent) and ankle under hips (reduced shin angle)
    - Hip drops significantly vs takeoff baseline
    """
    idx = mp_pose.PoseLandmark
    pts = {}
    for name in ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]:
        val = getattr(idx, name).value
        p = landmark_xy(landmarks, val, w, h)
        if not p:
            return False
        pts[name] = p

    # Average left/right for robustness
    hip_y = (pts["LEFT_HIP"][1] + pts["RIGHT_HIP"][1]) / 2
    knee_y = (pts["LEFT_KNEE"][1] + pts["RIGHT_KNEE"][1]) / 2
    ankle_y = (pts["LEFT_ANKLE"][1] + pts["RIGHT_ANKLE"][1]) / 2

    # Angles via triangle sides: knee angle (thigh-shin)
    def angle(a, b, c):
        # angle at b for triangle abc
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(max(min(cosang, 1), -1)))

    knee_angle_left = angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"])  # ~180 straight, smaller is squat
    knee_angle_right = angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"]) 
    knee_angle = (knee_angle_left + knee_angle_right) / 2

    # Heuristic thresholds
    is_knee_bent = knee_angle < 150  # <150° indicates noticeable squat
    is_hip_low = hip_y > (knee_y - 20)  # hips dropped to near knee height
    is_ankle_under = abs((pts["LEFT_ANKLE"][0] + pts["RIGHT_ANKLE"][0]) / 2 - (pts["LEFT_HIP"][0] + pts["RIGHT_HIP"][0]) / 2) < 60

    return is_knee_bent and (is_hip_low or is_ankle_under)


def main():
    cap = cv2.VideoCapture(0)
    # Lower resolution for faster, more stable real-time performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Enable OpenCV optimizations
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    state = "idle"  # idle -> armed -> airborne -> landed
    baseline_nose_y = None
    peak_delta_pixels = 0.0
    last_land_time = 0.0
    last_jump_result: Optional[JumpResult] = None

    with mp_pose.Pose(
        model_complexity=0,  # lighter model for real-time
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        prev_time = time.time()
        fps = 0.0
        # Buffer for smoothing nose signal
        nose_y_buffer: list[float] = []
        prev_dy: Optional[float] = None
        consec_up_frames = 0
        airborne_frames = 0
        dy_debug = 0.0
        vel_debug = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            # Process in RGB; reuse the same array to reduce copies
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            landmarks = None
            if res.pose_landmarks:
                landmarks = res.pose_landmarks.landmark
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Overlay last result
            if last_jump_result:
                text = f"Last: {'VALID' if last_jump_result.valid else 'BAD'} {last_jump_result.inches:.1f} in ({last_jump_result.reason})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if last_jump_result.valid else (0,0,255), 2)

            if not landmarks:
                cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if landmarks:
                nose_idx = mp_pose.PoseLandmark.NOSE.value
                right_hand_idx = mp_pose.PoseLandmark.RIGHT_INDEX.value
                left_hand_idx = mp_pose.PoseLandmark.LEFT_INDEX.value

                # Control gestures
                if state == "idle" and is_hand_to_nose(landmarks, w, h, right_hand_idx, nose_idx):
                    state = "armed"
                    # Set baseline nose height
                    p_nose = landmark_xy(landmarks, nose_idx, w, h)
                    if p_nose:
                        baseline_nose_y = p_nose[1]
                        peak_delta_pixels = 0.0
                    cv2.putText(frame, "Armed: ready to measure", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if state in ("armed", "landed") and is_hand_to_nose(landmarks, w, h, left_hand_idx, nose_idx):
                    # Reset
                    state = "idle"
                    baseline_nose_y = None
                    peak_delta_pixels = 0.0
                    last_jump_result = None

                # Measurement logic
                p_nose = landmark_xy(landmarks, nose_idx, w, h)
                if p_nose and baseline_nose_y is not None:
                    # Smooth nose y using small buffer
                    nose_y_buffer.append(p_nose[1])
                    if len(nose_y_buffer) > max(3, SMOOTHING_WINDOW):
                        nose_y_buffer.pop(0)
                    smooth_nose_y = sum(nose_y_buffer) / len(nose_y_buffer)

                    dy = baseline_nose_y - smooth_nose_y  # positive when moving up
                    dy_debug = dy
                    # Velocity (px/sec) of dy
                    now_t = time.time()
                    dt_local = now_t - prev_time if prev_time else 0.0
                    vel_dy = 0.0
                    if prev_dy is not None and dt_local > 0:
                        vel_dy = (dy - prev_dy) / dt_local
                    vel_debug = vel_dy
                    prev_dy = dy
                    peak_delta_pixels = max(peak_delta_pixels, dy)

                    # Determine state transitions using simple thresholds
                    arm_thresh = max(8, 0.015 * h)   # relative to frame height
                    land_thresh = max(4, 0.008 * h)
                    if state == "armed" and dy > arm_thresh:
                        state = "airborne"
                    elif state == "airborne" and dy < land_thresh:
                        up_vel_thresh = VEL_UP_PCT_OF_H * h
                        down_vel_thresh = VEL_DOWN_PCT_OF_H * h

                        if state == "armed":
                            if dy > arm_thresh and vel_dy > up_vel_thresh:
                                consec_up_frames += 1
                            else:
                                consec_up_frames = 0
                            if consec_up_frames >= 2:
                                state = "airborne"
                                airborne_frames = 0
                        elif state == "airborne":
                            airborne_frames += 1
                            if dy < land_thresh and vel_dy < -down_vel_thresh and airborne_frames >= MIN_AIRBORNE_FRAMES:
                                # Consider as landing moment
                                state = "landed"
                                last_land_time = time.time()
                                # Compute inches
                                inches = max(0.0, peak_delta_pixels) * inches_per_pixel()
                                if inches < MIN_JUMP_INCHES:
                                    last_jump_result = JumpResult(inches=inches, valid=False, reason="too-small")
                                else:
                                    cheated = detect_squat_cheat(landmarks, w, h)
                                    if cheated:
                                        last_jump_result = JumpResult(inches=inches, valid=False, reason="cheat")
                                    else:
                                        last_jump_result = JumpResult(inches=inches, valid=True, reason="ok")

                                # Prompt to reset with left hand
                                cv2.putText(frame, "Tap LEFT hand to nose to reset", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        # Consider as landing moment
                        state = "landed"
                        last_land_time = time.time()
                        # Compute inches
                        inches = max(0.0, peak_delta_pixels) * inches_per_pixel()
                        if inches < MIN_JUMP_INCHES:
                            last_jump_result = JumpResult(inches=inches, valid=False, reason="too-small")
                        else:
                            cheated = detect_squat_cheat(landmarks, w, h)
                            if cheated:
                                last_jump_result = JumpResult(inches=inches, valid=False, reason="cheat")
                            else:
                                last_jump_result = JumpResult(inches=inches, valid=True, reason="ok")

                        # Prompt to reset with left hand
                        cv2.putText(frame, "Tap LEFT hand to nose to reset", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # HUD for state
            cv2.putText(frame, f"State: {state}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Right hand -> nose: ARM", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Left hand -> nose: RESET", (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            # Debug values
            cv2.putText(frame, f"dy(px): {dy_debug:.1f}  vel(px/s): {vel_debug:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 2)

            # FPS display
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            cv2.imshow('Vertical Jump Measurement', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
