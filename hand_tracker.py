
import cv2
import mediapipe as mp
import math
import time

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"

MAX_HANDS = 1

PINCH_ON_THRESHOLD = 0.25
PINCH_OFF_THRESHOLD = 0.35

SMOOTHING_WINDOW = 5

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

SELECTION_THRESHOLD = 5.0

WEBCAM_INPUT_MIN = 0.15
WEBCAM_INPUT_MAX = 0.85

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

PLAYER_RADIUS = 9

TEAM_A_COLOR = (255, 100, 0)
TEAM_B_COLOR = (0, 140, 255)

SELECTED_COLOR = (0, 255, 0)
WHITE = (255, 255, 255)
GREEN = (40, 120, 40)
DARK_GREEN = (30, 100, 30)
BLUE = (255, 100, 0)
LIGHT_BLUE = (255, 200, 100)
RED = (0, 0, 255)


# ============================================================
# STATIC PLAYER POSITIONS
# 4-4-2 FORMATION
# ============================================================

PLAYERS = [
    # TEAM A: attacks left to right
    {"id": 1, "team": "A", "x": 5, "y": 34},
    {"id": 2, "team": "A", "x": 25, "y": 9},
    {"id": 3, "team": "A", "x": 22, "y": 25},
    {"id": 4, "team": "A", "x": 22, "y": 43},
    {"id": 5, "team": "A", "x": 25, "y": 59},
    {"id": 6, "team": "A", "x": 48, "y": 12},
    {"id": 7, "team": "A", "x": 45, "y": 27},
    {"id": 8, "team": "A", "x": 45, "y": 41},
    {"id": 9, "team": "A", "x": 48, "y": 56},
    {"id": 10, "team": "A", "x": 72, "y": 25},
    {"id": 11, "team": "A", "x": 72, "y": 43},

    # TEAM B: attacks right to left
    {"id": 12, "team": "B", "x": 100, "y": 34},
    {"id": 13, "team": "B", "x": 80, "y": 9},
    {"id": 14, "team": "B", "x": 83, "y": 25},
    {"id": 15, "team": "B", "x": 83, "y": 43},
    {"id": 16, "team": "B", "x": 80, "y": 59},
    {"id": 17, "team": "B", "x": 57, "y": 12},
    {"id": 18, "team": "B", "x": 60, "y": 27},
    {"id": 19, "team": "B", "x": 60, "y": 41},
    {"id": 20, "team": "B", "x": 57, "y": 56},
    {"id": 21, "team": "B", "x": 33, "y": 25},
    {"id": 22, "team": "B", "x": 33, "y": 43},
]


# ============================================================
# HAND TRACKING CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# HAND TRACKING FUNCTIONS
# ============================================================

def calculate_distance(point_a, point_b):
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


def calculate_pinch_ratio(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    pinch_distance = calculate_distance(thumb_tip, index_tip)
    reference_distance = calculate_distance(wrist, middle_mcp)

    if reference_distance == 0:
        return 0.0

    return pinch_distance / reference_distance


def update_pinch_state(ratio, current_state):
    if ratio < PINCH_ON_THRESHOLD:
        return True

    if ratio > PINCH_OFF_THRESHOLD:
        return False

    return current_state


def calculate_midpoint(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    midpoint_x = (thumb_tip.x + index_tip.x) / 2
    midpoint_y = (thumb_tip.y + index_tip.y) / 2

    return midpoint_x, midpoint_y


def smooth_midpoint(midpoint, history):
    history.append(midpoint)

    average_x = sum(point[0] for point in history) / len(history)
    average_y = sum(point[1] for point in history) / len(history)

    return average_x, average_y


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x = int((1 - landmark.x) * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw hand connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            BLUE,
            2
        )

    # Draw all landmarks
    for index, point in enumerate(points):
        color = LIGHT_BLUE if index == 8 else BLUE

        cv2.circle(
            frame,
            point,
            5,
            color,
            -1
        )


# ============================================================
# PITCH COORDINATE FUNCTIONS
# ============================================================

def get_pitch_bounds(frame):
    height, width, _ = frame.shape

    margin = 25

    left = margin
    right = width - margin
    top = margin
    bottom = height - margin

    return left, top, right, bottom


def pitch_to_pixel(x, y, bounds):
    left, top, right, bottom = bounds

    pixel_x = int(left + (x / PITCH_LENGTH) * (right - left))
    pixel_y = int(top + (y / PITCH_WIDTH) * (bottom - top))

    return pixel_x, pixel_y


def pixel_to_pitch(pixel_x, pixel_y, bounds):
    left, top, right, bottom = bounds

    x = ((pixel_x - left) / (right - left)) * PITCH_LENGTH
    y = ((pixel_y - top) / (bottom - top)) * PITCH_WIDTH

    return x, y


# ============================================================
# PITCH DRAWING
# ============================================================

def draw_pitch(frame):
    height, width, _ = frame.shape

    frame[:] = DARK_GREEN

    bounds = get_pitch_bounds(frame)
    left, top, right, bottom = bounds

    # Pitch background
    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        GREEN,
        -1
    )

    # Outer boundary
    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        WHITE,
        2
    )

    # Halfway line
    halfway_x = int((left + right) / 2)

    cv2.line(
        frame,
        (halfway_x, top),
        (halfway_x, bottom),
        WHITE,
        2
    )

    # Centre circle
    centre = pitch_to_pixel(
        PITCH_LENGTH / 2,
        PITCH_WIDTH / 2,
        bounds
    )

    centre_radius = int(
        (right - left) * 9.15 / PITCH_LENGTH
    )

    cv2.circle(
        frame,
        centre,
        centre_radius,
        WHITE,
        2
    )

    # Centre spot
    cv2.circle(frame, centre, 3, WHITE, -1)

    # Penalty areas
    penalty_length = 16.5
    penalty_width = 40.32

    six_yard_length = 5.5
    six_yard_width = 18.32

    # Left penalty area
    p1 = pitch_to_pixel(0, (PITCH_WIDTH - penalty_width) / 2, bounds)
    p2 = pitch_to_pixel(penalty_length, (PITCH_WIDTH + penalty_width) / 2, bounds)

    cv2.rectangle(frame, p1, p2, WHITE, 2)

    # Right penalty area
    p3 = pitch_to_pixel(
        PITCH_LENGTH - penalty_length,
        (PITCH_WIDTH - penalty_width) / 2,
        bounds
    )

    p4 = pitch_to_pixel(
        PITCH_LENGTH,
        (PITCH_WIDTH + penalty_width) / 2,
        bounds
    )

    cv2.rectangle(frame, p3, p4, WHITE, 2)

    # Left six-yard box
    s1 = pitch_to_pixel(0, (PITCH_WIDTH - six_yard_width) / 2, bounds)
    s2 = pitch_to_pixel(six_yard_length, (PITCH_WIDTH + six_yard_width) / 2, bounds)

    cv2.rectangle(frame, s1, s2, WHITE, 2)

    # Right six-yard box
    s3 = pitch_to_pixel(
        PITCH_LENGTH - six_yard_length,
        (PITCH_WIDTH - six_yard_width) / 2,
        bounds
    )

    s4 = pitch_to_pixel(
        PITCH_LENGTH,
        (PITCH_WIDTH + six_yard_width) / 2,
        bounds
    )

    cv2.rectangle(frame, s3, s4, WHITE, 2)

    # Penalty spots
    left_spot = pitch_to_pixel(11, PITCH_WIDTH / 2, bounds)
    right_spot = pitch_to_pixel(PITCH_LENGTH - 11, PITCH_WIDTH / 2, bounds)

    cv2.circle(frame, left_spot, 3, WHITE, -1)
    cv2.circle(frame, right_spot, 3, WHITE, -1)

    return bounds


# ============================================================
# PLAYER DRAWING
# ============================================================

def draw_players(frame, bounds, selected_player=None):
    for player in PLAYERS:
        position = pitch_to_pixel(
            player["x"],
            player["y"],
            bounds
        )

        if player["id"] == selected_player:
            cv2.circle(
                frame,
                position,
                PLAYER_RADIUS + 5,
                WHITE,
                2
            )

            cv2.circle(
                frame,
                position,
                PLAYER_RADIUS,
                SELECTED_COLOR,
                -1
            )

        else:
            color = (
                TEAM_A_COLOR
                if player["team"] == "A"
                else TEAM_B_COLOR
            )

            cv2.circle(
                frame,
                position,
                PLAYER_RADIUS,
                color,
                -1
            )

        cv2.putText(
            frame,
            str(player["id"]),
            (position[0] - 5, position[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            WHITE,
            1
        )


# ============================================================
# PLAYER SELECTION
# ============================================================

def find_nearest_player(cursor):
    nearest_player = None
    minimum_distance = float("inf")

    cursor_x, cursor_y = cursor

    for player in PLAYERS:
        distance = math.sqrt(
            (player["x"] - cursor_x) ** 2
            + (player["y"] - cursor_y) ** 2
        )

        if distance < minimum_distance:
            minimum_distance = distance
            nearest_player = player

    if minimum_distance <= SELECTION_THRESHOLD:
        return nearest_player["id"]

    return None


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    global PLAYERS

    # Check model file
    try:
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS
        )

        detector = vision.HandLandmarker.create_from_options(options)

    except Exception as error:
        print(f"Could not load hand tracking model: {error}")
        print(f"Check that the model exists at: {MODEL_PATH}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Single-hand state
    pinch_state = False
    midpoint_history = deque(maxlen=SMOOTHING_WINDOW)

    selected_player = None

    previous_time = time.time()
    timestamp_ms = 0

    print("Hand tracking and football pitch interface started.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Resize camera frame
        frame = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )

        height, width, _ = frame.shape

        # MediaPipe detection uses the unmirrored frame
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms += 33

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Mirror webcam display
        webcam_frame = cv2.flip(frame, 1)

        # Create pitch frame
        pitch_frame = frame.copy()

        pitch_bounds = draw_pitch(pitch_frame)

        selected_player = None

        # ====================================================
        # SINGLE HAND PROCESSING
        # ====================================================

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            draw_hand(webcam_frame, landmarks)

            # Pinch detection
            pinch_ratio = calculate_pinch_ratio(landmarks)

            pinch_state = update_pinch_state(
                pinch_ratio,
                pinch_state
            )

            # Smooth pinch midpoint
            midpoint = calculate_midpoint(landmarks)

            smoothed = smooth_midpoint(
                midpoint,
                midpoint_history
            )

            # Convert to mirrored webcam coordinates
            normalised_x = 1.0 - smoothed[0]
            normalised_y = smoothed[1]

            # Remap webcam input range to full pitch range
            remapped_x = (
                (normalised_x - WEBCAM_INPUT_MIN)
                / (WEBCAM_INPUT_MAX - WEBCAM_INPUT_MIN)
            )

            remapped_y = (
                (normalised_y - WEBCAM_INPUT_MIN)
                / (WEBCAM_INPUT_MAX - WEBCAM_INPUT_MIN)
            )

            # Clamp coordinates to 0-1
            remapped_x = max(0.0, min(1.0, remapped_x))
            remapped_y = max(0.0, min(1.0, remapped_y))

            # Convert directly to pitch coordinates
            pitch_cursor = (
                remapped_x * PITCH_LENGTH,
                remapped_y * PITCH_WIDTH
            )

            # Webcam cursor position
            webcam_x = int(normalised_x * width)
            webcam_y = int(normalised_y * height)

            cursor_color = (
                SELECTED_COLOR
                if pinch_state
                else RED
            )

            cv2.circle(
                webcam_frame,
                (webcam_x, webcam_y),
                10,
                cursor_color,
                -1
            )

            # Display pinch information
            cv2.putText(
                webcam_frame,
                f"Pinch Ratio: {pinch_ratio:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                webcam_frame,
                f"Pinch: {'ON' if pinch_state else 'OFF'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                cursor_color,
                2
            )

            # Select nearest player while pinching
            if pinch_state:
                selected_player = find_nearest_player(
                    pitch_cursor
                )

            # Draw pitch cursor
            pitch_cursor_pixel = pitch_to_pixel(
                pitch_cursor[0],
                pitch_cursor[1],
                pitch_bounds
            )

            cv2.circle(
                pitch_frame,
                pitch_cursor_pixel,
                10,
                SELECTED_COLOR if pinch_state else RED,
                2
            )

            cv2.putText(
                pitch_frame,
                f"Cursor: ({pitch_cursor[0]:.1f}, {pitch_cursor[1]:.1f})",
                (25, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1
            )

        # ====================================================
        # DRAW PLAYERS
        # ====================================================

        draw_players(
            pitch_frame,
            pitch_bounds,
            selected_player
        )

        # Selected player information
        if selected_player is not None:
            cv2.putText(
                pitch_frame,
                f"Selected Player: {selected_player}",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                SELECTED_COLOR,
                2
            )

        # ====================================================
        # FPS COUNTER
        # ====================================================

        current_time = time.time()
        elapsed_time = current_time - previous_time

        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        previous_time = current_time

        cv2.putText(
            webcam_frame,
            f"FPS: {int(fps)}",
            (width - 120, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )

        # ====================================================
        # DISPLAY BOTH INTERFACES
        # ====================================================

        combined_frame = cv2.hconcat([
            webcam_frame,
            pitch_frame
        ])

        cv2.imshow(
            "Hand Tracking Football Tactics Interface",
            combined_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Application closed.")


if __name__ == "__main__":
    main()
