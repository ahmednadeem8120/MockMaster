"""
mockmaster_demo_tracker.py — MockMaster Behavioural Tracking Demo
=================================================================
Standalone script for capturing demonstration screenshots of the
MediaPipe behavioural tracking system.

Opens a live OpenCV window showing:
  - MediaPipe FaceMesh landmark overlay (full 478-point mesh)
  - MediaPipe Hands landmark overlay
  - Real-time signal counters in the top-right corner panel
  - Per-frame signal status labels on the left (same as the API)
  - Running session metrics updated every 30 frames

HOW TO USE:
  1. Run:  python mockmaster_demo_tracker.py
  2. A window titled "MockMaster — Behavioural Tracking Demo" opens.
  3. Move around, gesture, smile, look away etc. to trigger different signals.
  4. Take screenshots at moments that capture different signals clearly:
       - Eye contact LOCKED vs BROKEN
       - Posture GOOD vs POOR (tilt head)
       - Enthusiasm HIGH (smile)
       - Hands ACTIVE (raise hand into frame)
       - Blink count incrementing
  5. Press Q to quit at any time.

REQUIREMENTS:
  pip install mediapipe opencv-python
  Python 3.11 or 3.12 required for mediapipe compatibility.
"""

import cv2
import time

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe opencv-python")
    exit(1)


# ---------------------------------------------------------------------------
# SIGNAL THRESHOLDS  (identical to behavioral_analyzer.py)
# ---------------------------------------------------------------------------
EYE_RATIO_LOW        = 0.35
EYE_RATIO_HIGH       = 0.65
POSTURE_X_THRESHOLD  = 0.06
NOSE_Y_LOW           = 0.40
NOSE_Y_HIGH          = 0.65
SMILE_RATIO_THRESH   = 0.40
BLINK_EAR_THRESHOLD  = 0.010
BLINK_DEBOUNCE       = 2        # consecutive frames required for a blink


# ---------------------------------------------------------------------------
# COLOURS
# ---------------------------------------------------------------------------
GREEN   = (0, 220, 100)
RED     = (0, 60, 220)
YELLOW  = (0, 220, 220)
WHITE   = (240, 240, 240)
ORANGE  = (0, 165, 255)
NAVY    = (100, 60, 20)
PANEL   = (30, 30, 30)


def draw_panel(image, metrics, elapsed):
    """Draw a dark metrics panel in the top-right corner."""
    h, w = image.shape[:2]
    panel_w = 280
    panel_h = 220
    x0 = w - panel_w - 10
    y0 = 10

    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Border
    cv2.rectangle(image, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

    # Title
    cv2.putText(image, "MOCKMASTER  LIVE METRICS",
                (x0 + 8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.line(image, (x0 + 8, y0 + 28), (x0 + panel_w - 8, y0 + 28), (80, 80, 80), 1)

    total = max(1, metrics["total_frames"])
    eye_pct  = round(metrics["eye_contact_frames"]  / total * 100, 1)
    post_pct = round(metrics["good_posture_frames"]  / total * 100, 1)
    enth_pct = round(metrics["smiling_frames"]       / total * 100, 1)
    gest_pct = round(metrics["hand_visible_frames"]  / total * 100, 1)
    blinks   = metrics["blink_count"]
    bpm      = round(blinks / max(1, elapsed) * 60, 1)

    rows = [
        ("Eye Contact",   f"{eye_pct}%",  GREEN  if eye_pct >= 55 else RED),
        ("Head Posture",  f"{post_pct}%", GREEN  if post_pct >= 50 else RED),
        ("Enthusiasm",    f"{enth_pct}%", YELLOW if enth_pct >= 8  else WHITE),
        ("Hand Gestures", f"{gest_pct}%", GREEN  if gest_pct >= 5  else ORANGE),
        ("Blinks",        f"{blinks}  ({bpm}/min)", WHITE),
        ("Elapsed",       f"{round(elapsed, 1)}s  |  {total} frames", (140, 140, 140)),
    ]

    for i, (label, value, colour) in enumerate(rows):
        y = y0 + 48 + i * 28
        cv2.putText(image, label + ":",
                    (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        cv2.putText(image, value,
                    (x0 + 130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1)

    return image


def draw_status_labels(image, signals):
    """Draw per-frame signal status labels on the left with a dark background panel."""
    labels = [
        (f"Eye Contact : {'LOCKED'  if signals['eye']     else 'BROKEN'}",   signals['eye'],     40),
        (f"Posture     : {'GOOD'    if signals['posture']  else 'POOR'}",     signals['posture'], 70),
        (f"Enthusiasm  : {'HIGH'    if signals['smile']    else 'NEUTRAL'}",  signals['smile'],   100),
        (f"Hands       : {'ACTIVE'  if signals['hands']    else 'HIDDEN'}",   signals['hands'],   130),
        (f"Blinks      : {signals['blinks']}",                                 True,               160),
    ]

    # --- Semi-transparent dark background panel behind the labels ---
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 16), (310, 175), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.rectangle(image, (8, 16), (310, 175), (80, 80, 80), 1)

    # --- Text labels ---
    for text, active, y in labels:
        colour = GREEN if active else (WHITE if "Blinks" in text else RED)
        if "Enthusiasm" in text and not active:
            colour = WHITE
        if "Hands" in text and not active:
            colour = ORANGE
        cv2.putText(image, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 2)


def draw_mesh(image, face_landmarks, mp_drawing, mp_face_mesh):
    """Draw the full FaceMesh landmark overlay."""
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 180, 80), thickness=1, circle_radius=0)
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 220, 120), thickness=1, circle_radius=0)
    )


def run():
    mp_drawing   = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands_sol = mp.solutions.hands
    mp_hands_draw = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands_tracker = mp_hands_sol.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("MockMaster Demo Tracker running.")
    print("Press Q to quit.")
    print()
    print("Suggested screenshots:")
    print("  1. Normal forward-facing — Eye LOCKED, Posture GOOD")
    print("  2. Look away / tilt head — Eye BROKEN or Posture POOR")
    print("  3. Smile broadly          — Enthusiasm HIGH")
    print("  4. Raise hand into frame  — Hands ACTIVE")

    metrics = {
        "total_frames":       0,
        "eye_contact_frames": 0,
        "good_posture_frames":0,
        "smiling_frames":     0,
        "hand_visible_frames":0,
        "blink_count":        0,
    }

    is_blinking       = False
    blink_frame_counter = 0
    start_time        = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)

        # ── Crop to square using the shorter dimension ──────────────────────
        h, w = image.shape[:2]
        side = min(h, w)
        x0_crop = (w - side) // 2
        y0_crop = (h - side) // 2
        image = image[y0_crop:y0_crop + side, x0_crop:x0_crop + side]
        # ────────────────────────────────────────────────────────────────────

        metrics["total_frames"] += 1
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results  = face_mesh.process(image_rgb)
        hand_results  = hands_tracker.process(image_rgb)
        elapsed       = time.time() - start_time

        signals = {
            "eye":    False,
            "posture":False,
            "smile":  False,
            "hands":  False,
            "blinks": metrics["blink_count"],
        }

        # --- Hand tracking ---
        if hand_results.multi_hand_landmarks:
            signals["hands"] = True
            metrics["hand_visible_frames"] += 1
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_lm,
                    mp_hands_sol.HAND_CONNECTIONS,
                    mp_hands_draw.get_default_hand_landmarks_style(),
                    mp_hands_draw.get_default_hand_connections_style()
                )

        # --- Face tracking ---
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:

                draw_mesh(image, face_landmarks, mp_drawing, mp_face_mesh)

                lm = face_landmarks.landmark

                # Eye contact
                left_outer  = lm[33].x
                left_inner  = lm[133].x
                pupil       = lm[468].x
                eye_width   = left_inner - left_outer
                if eye_width != 0:
                    ratio = (pupil - left_outer) / eye_width
                    if EYE_RATIO_LOW < ratio < EYE_RATIO_HIGH:
                        signals["eye"] = True
                        metrics["eye_contact_frames"] += 1

                # Head posture
                nose_x       = lm[1].x
                nose_y       = lm[1].y
                chin_y       = lm[152].y
                forehead_y   = lm[10].y
                left_cheek   = lm[234].x
                right_cheek  = lm[454].x
                face_cx      = (left_cheek + right_cheek) / 2
                face_h       = chin_y - forehead_y
                nose_rel_y   = (nose_y - forehead_y) / face_h if face_h else 0.5
                forward      = abs(nose_x - face_cx) < POSTURE_X_THRESHOLD
                head_up      = NOSE_Y_LOW < nose_rel_y < NOSE_Y_HIGH
                if forward and head_up:
                    signals["posture"] = True
                    metrics["good_posture_frames"] += 1

                # Smile
                mouth_l  = lm[61].x
                mouth_r  = lm[291].x
                mw       = mouth_r - mouth_l
                fw       = right_cheek - left_cheek
                if fw and (mw / fw) > SMILE_RATIO_THRESH:
                    signals["smile"] = True
                    metrics["smiling_frames"] += 1

                # Blink
                top_lid    = lm[159].y
                bot_lid    = lm[145].y
                ear        = bot_lid - top_lid
                if ear < BLINK_EAR_THRESHOLD:
                    blink_frame_counter += 1
                    if blink_frame_counter >= BLINK_DEBOUNCE and not is_blinking:
                        metrics["blink_count"] += 1
                        signals["blinks"] = metrics["blink_count"]
                        is_blinking = True
                else:
                    blink_frame_counter = 0
                    is_blinking = False

        # --- Overlays ---
        draw_status_labels(image, signals)
        draw_panel(image, metrics, elapsed)

        # Window title hint
        cv2.putText(image,
                    "MockMaster — Behavioural Tracking Demo  |  Press Q to quit",
                    (10, image.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.imshow("MockMaster — Behavioural Tracking Demo", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo tracker closed.")


if __name__ == "__main__":
    run()