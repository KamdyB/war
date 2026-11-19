
import cv2
import mediapipe as mp
import time
import math
from pathlib import Path
from collections import deque

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
MAX_HANDS = 2

# Pinch detection thresholds
PINCH_ON_THRESHOLD = 0.25
PINCH_OFF_THRESHOLD = 0.35

# Number of frames used for midpoint smoothing
SMOOTHING_WINDOW = 5

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Drawing colours (BGR)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
LANDMARK_COLOUR = (255, 180, 50)
CONNECTION_COLOUR = (80, 220, 120)


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    print("Download hand_landmarker.task into the models/ folder.")
    raise SystemExit(1)


# --------------------------------------------------
# MEDIAPIPE INITIALIZATION
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# PINCH DETECTION
# --------------------------------------------------

def calculate_distance(point_a, point_b):
    """Calculate Euclidean distance between two 3D landmarks."""
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


def calculate_pinch_ratio(landmarks):
    """
    Pinch ratio =
    distance between thumb tip (4) and index tip (8)
    divided by distance between wrist (0) and middle
    finger MCP (9).
    """

    pinch_distance = calculate_distance(
        landmarks[4],
        landmarks[8]
    )

    reference_distance = calculate_distance(
        landmarks[0],
        landmarks[9]
    )

    if reference_distance == 0:
        return None

    return pinch_distance / reference_distance


def update_pinch_state(ratio, currently_pinched):
    """
    Hysteresis prevents rapid switching around
    the pinch threshold.
    """

    if ratio is None:
        return False

    if not currently_pinched and ratio < PINCH_ON_THRESHOLD:
        return True

    if currently_pinched and ratio > PINCH_OFF_THRESHOLD:
        return False

    return currently_pinched


# --------------------------------------------------
# MIDPOINT SMOOTHING
# --------------------------------------------------

def calculate_midpoint(landmarks):
    """Find the midpoint between thumb and index fingertips."""

    thumb = landmarks[4]
    index = landmarks[8]

    return (
        (thumb.x + index.x) / 2,
        (thumb.y + index.y) / 2
    )


def smooth_midpoint(midpoint, history):
    """Return the moving average of the last five midpoints."""

    history.append(midpoint)

    average_x = sum(point[0] for point in history) / len(history)
    average_y = sum(point[1] for point in history) / len(history)

    return average_x, average_y


# --------------------------------------------------
# DRAW HAND LANDMARKS
# --------------------------------------------------

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x = int((1.0 - landmark.x) * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            CONNECTION_COLOUR,
            2
        )

    # Draw landmarks
    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            LANDMARK_COLOUR,
            -1
        )

    return points


# --------------------------------------------------
# MAIN CAMERA LOOP
# --------------------------------------------------

def main():

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_index = 0
    start_time = time.perf_counter()
    fps = 0.0

    # Maintain independent pinch state and smoothing
    # history for each detected hand.
    pinch_states = {}
    midpoint_histories = {}

    print("Hand tracker with pinch detection started.")
    print("Press 'q' to quit.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Detect on original frame
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = frame_index * 33
                frame_index += 1

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # Mirror display only
                display_frame = cv2.flip(frame, 1)

                hand_count = len(result.hand_landmarks)

                active_hands = set()

                for hand_id, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    active_hands.add(hand_id)

                    # Draw skeleton
                    draw_hand(display_frame, hand_landmarks)

                    # Initialize state for newly detected hands
                    if hand_id not in pinch_states:
                        pinch_states[hand_id] = False
                        midpoint_histories[hand_id] = deque(
                            maxlen=SMOOTHING_WINDOW
                        )

                    # Calculate pinch ratio
                    ratio = calculate_pinch_ratio(hand_landmarks)

                    # Update hysteresis state
                    pinch_states[hand_id] = update_pinch_state(
                        ratio,
                        pinch_states[hand_id]
                    )

                    is_pinched = pinch_states[hand_id]

                    # Calculate and smooth midpoint
                    midpoint = calculate_midpoint(hand_landmarks)

                    smoothed_midpoint = smooth_midpoint(
                        midpoint,
                        midpoint_histories[hand_id]
                    )

                    # Convert normalized coordinates to pixels.
                    # Mirror x to match the displayed frame.
                    height, width, _ = display_frame.shape

                    midpoint_x = int(
                        (1.0 - smoothed_midpoint[0]) * width
                    )

                    midpoint_y = int(
                        smoothed_midpoint[1] * height
                    )

                    # Green when pinched, red otherwise
                    circle_colour = GREEN if is_pinched else RED

                    cv2.circle(
                        display_frame,
                        (midpoint_x, midpoint_y),
                        12,
                        circle_colour,
                        -1
                    )

                    # Display pinch ratio
                    ratio_text = (
                        f"Hand {hand_id + 1} Ratio: {ratio:.3f}"
                        if ratio is not None
                        else f"Hand {hand_id + 1} Ratio: N/A"
                    )

                    cv2.putText(
                        display_frame,
                        ratio_text,
                        (20, 145 + hand_id * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        WHITE,
                        2
                    )

                    # Display pinch status
                    status = "PINCHED" if is_pinched else "NOT PINCHED"

                    cv2.putText(
                        display_frame,
                        f"Hand {hand_id + 1}: {status}",
                        (20, 180 + hand_id * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        circle_colour,
                        2
                    )

                # Remove stale states for hands no longer detected
                for hand_id in list(pinch_states):
                    if hand_id not in active_hands:
                        del pinch_states[hand_id]
                        del midpoint_histories[hand_id]

                # Calculate FPS
                elapsed = time.perf_counter() - start_time

                if elapsed > 0:
                    fps = frame_index / elapsed

                # Display general information
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    GREEN,
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Hands detected: {hand_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    WHITE,
                    2
                )

                cv2.putText(
                    display_frame,
                    "Press Q to quit",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    WHITE,
                    1
                )

                cv2.imshow(
                    "Hand Tracking and Pinch Detection",
                    display_frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Hand tracker stopped.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
