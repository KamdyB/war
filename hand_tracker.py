
import cv2
import mediapipe as mp
import time
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
MAX_HANDS = 2

# Hand landmark connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Little finger
    (0, 17)                                # Palm
]


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
# DRAW HAND LANDMARKS
# --------------------------------------------------

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        # Coordinates are normalized between 0 and 1.
        # Account for the mirrored display.
        x = int((1.0 - landmark.x) * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections first.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (80, 220, 120),
            2
        )

    # Draw all 21 landmarks.
    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            (255, 180, 50),
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
        print("Check your camera connection or try CAMERA_INDEX = 1.")
        return

    frame_index = 0
    start_time = time.perf_counter()
    fps = 0.0

    print("Hand tracker started.")
    print("Press 'q' to quit.")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = camera.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                # Detect on the original, unmirrored frame.
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Video mode requires monotonically increasing
                # timestamps in milliseconds.
                timestamp_ms = frame_index * 33
                frame_index += 1

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # Mirror only the display.
                display_frame = cv2.flip(frame, 1)

                # Draw detected hands.
                hand_count = len(result.hand_landmarks)

                for hand_landmarks in result.hand_landmarks:
                    draw_hand(display_frame, hand_landmarks)

                # Calculate FPS.
                elapsed = time.perf_counter() - start_time

                if elapsed > 0:
                    fps = frame_index / elapsed

                # Display information.
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Hands detected: {hand_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    "Press Q to quit",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1
                )

                cv2.imshow(
                    "Live Hand Tracking",
                    display_frame
                )

                # Quit when Q is pressed.
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
