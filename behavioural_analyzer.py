import cv2
import json
import time
import threading

# --- Session control ---------------------------------------------------------
# `stop_event` lets the API layer signal the video generator to break out of
# its capture loop when the interview ends. Without this, the generator keeps
# holding the webcam and the `behavioral_metrics.json` that `/end` reads is
# mid-session data, not the final counts.
#
# `session_start_ts` is set when tracking begins and lets us normalise the
# blink rate by actual elapsed seconds instead of question count.
stop_event = threading.Event()
session_start_ts = {"value": None}

# Path used by every read/write so the API layer uses the same file.
METRICS_PATH = "behavioral_metrics.json"


def reset_session():
    """Called by /start so a new interview begins with a clean slate."""
    stop_event.clear()
    session_start_ts["value"] = time.time()


def stop_session_and_wait(timeout: float = 2.0):
    """
    Called by /end. Signals the generator to exit, then polls the metrics file
    until it shows `status: completed` (or the timeout expires). This guarantees
    /end reads finalised numbers, not a mid-session snapshot.
    """
    stop_event.set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            if data.get("status") == "completed":
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    # Probe the attribute that fails on Python 3.14
    _ = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
    print("Mediapipe loaded successfully — full behavioral tracking enabled.")
except (ImportError, AttributeError) as e:
    print(f"WARNING: Mediapipe not available on this Python version ({e}).")
    print("Video feed will stream without behavioral analysis overlays.")
    print("To enable full tracking, use Python 3.11 or 3.12.")
 
 
def _generate_frames_with_tracking():
    """Full tracking path — only runs when mediapipe is available."""
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
 
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands_tracker = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
 
    cap = cv2.VideoCapture(0)
 
    total_frames = 0
    eye_contact_frames = 0
    good_posture_frames = 0
    smiling_frames = 0
    hand_visible_frames = 0
    blink_count = 0
    is_blinking = False
    blink_frame_counter = 0   # consecutive frames the eye has been below threshold

    # Mark the start of the tracking session if /start hasn't already done so.
    # Lets us normalise blink rate by elapsed seconds, not question count.
    if session_start_ts["value"] is None:
        session_start_ts["value"] = time.time()
 
    print("Behavioral Analysis Online: Tracking Face & Hands.")
 
    try:
        while cap.isOpened():
            # Exit as soon as /end signals the session is over, so the webcam
            # is released and the final metrics JSON is written immediately.
            if stop_event.is_set():
                print("Stop event received — ending behavioral tracking.")
                break

            success, image = cap.read()
            if not success:
             time.sleep(0.1)
             continue
            image = cv2.flip(image, 1)
            total_frames += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
            face_results = face_mesh.process(image_rgb)
            hand_results = hands_tracker.process(image_rgb)
 
            # --- Hand presence ---
            is_using_hands = False
            if hand_results.multi_hand_landmarks:
                is_using_hands = True
                hand_visible_frames += 1
 
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
 
                    # Eye contact (iris tracking)
                    left_eye_outer = face_landmarks.landmark[33].x
                    left_eye_inner = face_landmarks.landmark[133].x
                    pupil_center = face_landmarks.landmark[468].x
                    eye_width = left_eye_inner - left_eye_outer
                    is_eye_contact = False
                    if eye_width != 0:
                        pupil_ratio = (pupil_center - left_eye_outer) / eye_width
                        if 0.35 < pupil_ratio < 0.65:
                            is_eye_contact = True
                            eye_contact_frames += 1
 
                    # Head posture
                    nose_x = face_landmarks.landmark[1].x
                    nose_y = face_landmarks.landmark[1].y
                    chin_y = face_landmarks.landmark[152].y
                    forehead_y = face_landmarks.landmark[10].y
                    left_cheek_x = face_landmarks.landmark[234].x
                    right_cheek_x = face_landmarks.landmark[454].x
                    face_center_x = (left_cheek_x + right_cheek_x) / 2
                    is_looking_forward = abs(nose_x - face_center_x) < 0.06
                    face_height = chin_y - forehead_y
                    nose_relative_y = (nose_y - forehead_y) / face_height if face_height != 0 else 0.5
                    is_head_up = 0.4 < nose_relative_y < 0.65
                    is_good_posture = is_looking_forward and is_head_up
                    if is_good_posture:
                        good_posture_frames += 1
 
                    # Smile detection
                    mouth_left = face_landmarks.landmark[61].x
                    mouth_right = face_landmarks.landmark[291].x
                    mouth_width = mouth_right - mouth_left
                    face_width = right_cheek_x - left_cheek_x
                    is_smiling = False
                    if face_width != 0:
                        smile_ratio = mouth_width / face_width
                        # Landmarks 61/291 are inner mouth corners, not outer lip edges,
                        # so the measured width is narrower than the true smile width.
                        # 0.40 is the correct threshold for detecting a genuine smile.
                        # (The old 0.50 was unreachable — no frames were ever counted.)
                        if smile_ratio > 0.40:
                            is_smiling = True
                            smiling_frames += 1
 
                    # Blink rate — two fixes applied:
                    # 1. Threshold lowered from 0.014 → 0.010. The old value was too
                    #    loose and fired on normal squinting / downward glances.
                    # 2. Added a 2-frame debounce (blink_frame_counter). A genuine blink
                    #    closes the eye for ~100-200ms (~3-6 frames at 30fps). Requiring
                    #    2 consecutive below-threshold frames filters out single-frame
                    #    noise and eyelid bounce that caused massive overcounting.
                    top_eyelid = face_landmarks.landmark[159].y
                    bottom_eyelid = face_landmarks.landmark[145].y
                    eye_open_distance = bottom_eyelid - top_eyelid
                    if eye_open_distance < 0.010:
                        blink_frame_counter += 1
                        if blink_frame_counter >= 2 and not is_blinking:
                            blink_count += 1
                            is_blinking = True
                    else:
                        blink_frame_counter = 0
                        is_blinking = False
 
                    # Overlay text
                    cv2.putText(image, f"Eye Contact: {'LOCKED' if is_eye_contact else 'BROKEN'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_eye_contact else (0, 0, 255), 2)
                    cv2.putText(image, f"Posture: {'GOOD' if is_good_posture else 'POOR'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_good_posture else (0, 0, 255), 2)
                    cv2.putText(image, f"Enthusiasm: {'HIGH' if is_smiling else 'NEUTRAL'}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0) if is_smiling else (255, 255, 255), 2)
                    cv2.putText(image, f"Hands: {'ACTIVE' if is_using_hands else 'HIDDEN'}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_using_hands else (255, 165, 0), 2)
                    cv2.putText(image, f"Blinks: {blink_count}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
            # Save metrics every ~30 frames
            if total_frames > 0 and total_frames % 30 == 0:
                elapsed = max(1.0, time.time() - (session_start_ts["value"] or time.time()))
                metrics = {
                    "eye_contact_score": round((eye_contact_frames / total_frames) * 100, 1),
                    "head_posture_score": round((good_posture_frames / total_frames) * 100, 1),
                    "enthusiasm_score": round((smiling_frames / total_frames) * 100, 1),
                    "hand_gesture_score": round((hand_visible_frames / total_frames) * 100, 1),
                    "blink_count": blink_count,
                    "session_duration_sec": round(elapsed, 1),
                    "status": "recording"
                }
                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f)
 
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
 
    finally:
        cap.release()
        elapsed = max(1.0, time.time() - (session_start_ts["value"] or time.time()))
        final_metrics = {
            "eye_contact_score": round((eye_contact_frames / max(1, total_frames)) * 100, 1),
            "head_posture_score": round((good_posture_frames / max(1, total_frames)) * 100, 1),
            "enthusiasm_score": round((smiling_frames / max(1, total_frames)) * 100, 1),
            "hand_gesture_score": round((hand_visible_frames / max(1, total_frames)) * 100, 1),
            "blink_count": blink_count,
            "session_duration_sec": round(elapsed, 1),
            "status": "completed"
        }
        with open(METRICS_PATH, "w") as f:
            json.dump(final_metrics, f)
        print("Behavioral tracking stopped. Final metrics saved.")
 
 
def _generate_frames_plain():
    """
    Fallback path when mediapipe is unavailable.
    Streams raw webcam frames with a warning overlay so the app
    continues to function. Behavioral scores will all be 0 in the report.
    """
    cap = cv2.VideoCapture(0)
    print("Plain video feed active (no behavioral analysis).")

    if session_start_ts["value"] is None:
        session_start_ts["value"] = time.time()

    # Write zeroed metrics so /end doesn't crash when reading the file
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "eye_contact_score": 0,
            "head_posture_score": 0,
            "enthusiasm_score": 0,
            "hand_gesture_score": 0,
            "blink_count": 0,
            "session_duration_sec": 0,
            "status": "recording",
            "warning": "Behavioral tracking unavailable — mediapipe requires Python 3.11 or 3.12"
        }, f)
 
    try:
        while cap.isOpened():
            if stop_event.is_set():
                print("Stop event received — ending plain video feed.")
                break

            success, image = cap.read()
            if not success:
             time.sleep(0.1)
             continue
            image = cv2.flip(image, 1)
 
            cv2.putText(
                image,
                "Tracking unavailable (Python 3.14)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 165, 255), 2
            )
            cv2.putText(
                image,
                "Use Python 3.11/3.12 to enable AI tracking",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1
            )
 
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        # Mark the file as completed so /end doesn't hang waiting for it.
        elapsed = max(1.0, time.time() - (session_start_ts["value"] or time.time()))
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "eye_contact_score": 0,
                "head_posture_score": 0,
                "enthusiasm_score": 0,
                "hand_gesture_score": 0,
                "blink_count": 0,
                "session_duration_sec": round(elapsed, 1),
                "status": "completed",
                "warning": "Behavioral tracking unavailable — mediapipe requires Python 3.11 or 3.12"
            }, f)
        print("Plain video feed stopped.")
 
 
def generate_video_frames():
    """
    Public entry point called by api.py.
    Routes to full tracking or plain feed based on mediapipe availability.
    """
    if MEDIAPIPE_AVAILABLE:
        yield from _generate_frames_with_tracking()
    else:
        yield from _generate_frames_plain()