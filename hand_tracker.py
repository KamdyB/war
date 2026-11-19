
import argparse
import math
import time
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from collections import deque
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MAX_HANDS = 1

# Pinch detection
PINCH_ON_THRESHOLD = 0.25
PINCH_OFF_THRESHOLD = 0.35
SMOOTHING_WINDOW = 5

# Webcam-to-dashboard mapping
WEBCAM_INPUT_MIN = 0.15
WEBCAM_INPUT_MAX = 0.85

# Dashboard dimensions
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
WINDOW_NAME = "CSV Gesture Dashboard"

# Dashboard layout
DASHBOARD_MARGIN = 30
TABLE_WIDTH = 680
PANEL_GAP = 30
RIGHT_PANEL_WIDTH = 510

TABLE_X = DASHBOARD_MARGIN
TABLE_Y = 145
TABLE_HEADER_HEIGHT = 65
TABLE_ROW_HEIGHT = 42
TABLE_VISIBLE_ROWS = 12
MAX_DISPLAY_COLUMNS = 6

CHART_X = TABLE_X + TABLE_WIDTH + PANEL_GAP
CHART_Y = 145
CHART_WIDTH = RIGHT_PANEL_WIDTH
CHART_HEIGHT = 400
CHART_MAX_ROWS = 10

# Webcam preview
PREVIEW_WIDTH = WINDOW_WIDTH // 4
PREVIEW_HEIGHT = WINDOW_HEIGHT // 4
PREVIEW_MARGIN = 20

# Hand drawing
HAND_CONNECTION_COLOR = (190, 150, 90)
HAND_LANDMARK_COLOR = (220, 190, 130)
INDEX_TIP_COLOR = (255, 220, 100)
HAND_LINE_THICKNESS = 2
HAND_POINT_RADIUS = 4

# Dashboard colours: BGR
BACKGROUND = (24, 27, 35)
PANEL_BACKGROUND = (33, 38, 49)
HEADER_BACKGROUND = (45, 53, 68)
BUTTON_BACKGROUND = (54, 65, 83)
BUTTON_SELECTED = (35, 115, 95)
ACCENT_COLOR = (90, 210, 170)
TEXT_COLOR = (235, 238, 242)
MUTED_TEXT = (145, 155, 170)
GRID_COLOR = (58, 66, 80)
BAR_COLOR = (90, 190, 150)
CURSOR_COLOR = (80, 240, 190)
PINCH_COLOR = (50, 255, 100)
WARNING_COLOR = (80, 150, 255)

# Text
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_FONT_SCALE = 0.85
SUBTITLE_FONT_SCALE = 0.48
BODY_FONT_SCALE = 0.45
HEADER_FONT_SCALE = 0.43
SMALL_FONT_SCALE = 0.38

# Interaction thresholds
SCROLL_PIXELS_PER_ROW = 42
MAX_SCROLL_STEP = 5


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
# HAND TRACKING
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

    pinch_distance = calculate_distance(
        thumb_tip,
        index_tip
    )

    reference_distance = calculate_distance(
        wrist,
        middle_mcp
    )

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

    return (
        (thumb_tip.x + index_tip.x) / 2,
        (thumb_tip.y + index_tip.y) / 2
    )


def smooth_midpoint(midpoint, history):
    history.append(midpoint)

    average_x = sum(point[0] for point in history) / len(history)
    average_y = sum(point[1] for point in history) / len(history)

    return average_x, average_y


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int((1.0 - landmark.x) * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            HAND_CONNECTION_COLOR,
            HAND_LINE_THICKNESS
        )

    for index, point in enumerate(points):
        color = (
            INDEX_TIP_COLOR
            if index == 8
            else HAND_LANDMARK_COLOR
        )

        cv2.circle(
            frame,
            point,
            HAND_POINT_RADIUS,
            color,
            -1
        )


# ============================================================
# COORDINATE MAPPING
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def map_webcam_to_dashboard(normalised_x, normalised_y):
    remapped_x = (
        (normalised_x - WEBCAM_INPUT_MIN)
        / (WEBCAM_INPUT_MAX - WEBCAM_INPUT_MIN)
    )

    remapped_y = (
        (normalised_y - WEBCAM_INPUT_MIN)
        / (WEBCAM_INPUT_MAX - WEBCAM_INPUT_MIN)
    )

    remapped_x = clamp(remapped_x, 0.0, 1.0)
    remapped_y = clamp(remapped_y, 0.0, 1.0)

    dashboard_x = int(remapped_x * WINDOW_WIDTH)
    dashboard_y = int(remapped_y * WINDOW_HEIGHT)

    return dashboard_x, dashboard_y


# ============================================================
# DASHBOARD LAYOUT
# ============================================================

def draw_dashboard_background(frame):
    frame[:] = BACKGROUND

    cv2.putText(
        frame,
        "DATA EXPLORER",
        (DASHBOARD_MARGIN, 55),
        FONT,
        TITLE_FONT_SCALE,
        TEXT_COLOR,
        2
    )

    cv2.putText(
        frame,
        "Gesture-controlled CSV dashboard",
        (DASHBOARD_MARGIN, 85),
        FONT,
        SUBTITLE_FONT_SCALE,
        MUTED_TEXT,
        1
    )

    cv2.putText(
        frame,
        "PINCH A COLUMN HEADER TO SELECT",
        (DASHBOARD_MARGIN, 115),
        FONT,
        SMALL_FONT_SCALE,
        ACCENT_COLOR,
        1
    )


# ============================================================
# TABLE
# ============================================================

def get_display_columns(dataframe):
    return list(dataframe.columns[:MAX_DISPLAY_COLUMNS])


def get_column_widths(columns):
    if not columns:
        return []

    base_width = TABLE_WIDTH // len(columns)
    widths = [base_width] * len(columns)

    remainder = TABLE_WIDTH - sum(widths)

    for index in range(remainder):
        widths[index] += 1

    return widths


def get_header_regions(columns):
    widths = get_column_widths(columns)
    regions = []

    current_x = TABLE_X

    for index, column in enumerate(columns):
        width = widths[index]

        regions.append({
            "column": column,
            "x1": current_x,
            "y1": TABLE_Y,
            "x2": current_x + width,
            "y2": TABLE_Y + TABLE_HEADER_HEIGHT
        })

        current_x += width

    return regions


def draw_table(frame, dataframe, selected_column, scroll_offset):
    columns = get_display_columns(dataframe)

    if not columns:
        cv2.putText(
            frame,
            "No columns available.",
            (TABLE_X, TABLE_Y + 40),
            FONT,
            BODY_FONT_SCALE,
            WARNING_COLOR,
            1
        )
        return []

    widths = get_column_widths(columns)
    header_regions = get_header_regions(columns)

    # Table panel
    cv2.rectangle(
        frame,
        (TABLE_X - 5, TABLE_Y - 5),
        (
            TABLE_X + TABLE_WIDTH,
            TABLE_Y + TABLE_HEADER_HEIGHT
            + TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT
        ),
        PANEL_BACKGROUND,
        -1
    )

    # Column header buttons
    current_x = TABLE_X

    for index, column in enumerate(columns):
        width = widths[index]

        is_selected = column == selected_column

        button_color = (
            BUTTON_SELECTED
            if is_selected
            else BUTTON_BACKGROUND
        )

        cv2.rectangle(
            frame,
            (current_x + 2, TABLE_Y + 2),
            (
                current_x + width - 2,
                TABLE_Y + TABLE_HEADER_HEIGHT - 2
            ),
            button_color,
            -1
        )

        label = str(column)

        cv2.putText(
            frame,
            label[:16],
            (current_x + 9, TABLE_Y + 38),
            FONT,
            HEADER_FONT_SCALE,
            TEXT_COLOR,
            1
        )

        current_x += width

    # Table rows
    visible_data = dataframe.iloc[
        scroll_offset:scroll_offset + TABLE_VISIBLE_ROWS
    ]

    for visible_index, (_, row) in enumerate(visible_data.iterrows()):
        row_y = (
            TABLE_Y
            + TABLE_HEADER_HEIGHT
            + visible_index * TABLE_ROW_HEIGHT
        )

        if visible_index % 2 == 0:
            cv2.rectangle(
                frame,
                (TABLE_X, row_y),
                (
                    TABLE_X + TABLE_WIDTH,
                    row_y + TABLE_ROW_HEIGHT
                ),
                (38, 44, 56),
                -1
            )

        current_x = TABLE_X

        for column_index, column in enumerate(columns):
            value = row[column]

            if pd.isna(value):
                display_value = "-"
            else:
                display_value = str(value)

            cv2.putText(
                frame,
                display_value[:16],
                (current_x + 9, row_y + 27),
                FONT,
                BODY_FONT_SCALE,
                TEXT_COLOR,
                1
            )

            current_x += widths[column_index]

        cv2.line(
            frame,
            (TABLE_X, row_y + TABLE_ROW_HEIGHT),
            (
                TABLE_X + TABLE_WIDTH,
                row_y + TABLE_ROW_HEIGHT
            ),
            GRID_COLOR,
            1
        )

    # Table information
    cv2.putText(
        frame,
        f"Rows {scroll_offset + 1}-"
        f"{min(scroll_offset + TABLE_VISIBLE_ROWS, len(dataframe))}"
        f" of {len(dataframe)}",
        (TABLE_X, TABLE_Y + TABLE_HEADER_HEIGHT
         + TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT + 25),
        FONT,
        SMALL_FONT_SCALE,
        MUTED_TEXT,
        1
    )

    return header_regions


# ============================================================
# CHART
# ============================================================

def get_numeric_columns(dataframe):
    numeric_columns = []

    for column in dataframe.columns:
        converted = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(column)

    return numeric_columns


def draw_chart(frame, dataframe, selected_column):
    panel_x = CHART_X
    panel_y = CHART_Y

    cv2.rectangle(
        frame,
        (panel_x - 10, panel_y - 10),
        (
            panel_x + CHART_WIDTH,
            panel_y + CHART_HEIGHT
        ),
        PANEL_BACKGROUND,
        -1
    )

    cv2.putText(
        frame,
        "COLUMN ANALYSIS",
        (panel_x + 10, panel_y + 30),
        FONT,
        SUBTITLE_FONT_SCALE,
        MUTED_TEXT,
        1
    )

    if selected_column is None:
        message = "Select a column header"

        cv2.putText(
            frame,
            message,
            (panel_x + 20, panel_y + 100),
            FONT,
            BODY_FONT_SCALE,
            WARNING_COLOR,
            1
        )
        return

    numeric_columns = get_numeric_columns(dataframe)

    if selected_column not in numeric_columns:
        cv2.putText(
            frame,
            f"{selected_column}",
            (panel_x + 10, panel_y + 70),
            FONT,
            0.65,
            TEXT_COLOR,
            1
        )

        cv2.putText(
            frame,
            "This column is not numeric.",
            (panel_x + 10, panel_y + 110),
            FONT,
            BODY_FONT_SCALE,
            WARNING_COLOR,
            1
        )
        return

    cv2.putText(
        frame,
        str(selected_column)[:30],
        (panel_x + 10, panel_y + 75),
        FONT,
        0.65,
        ACCENT_COLOR,
        1
    )

    values = pd.to_numeric(
        dataframe[selected_column],
        errors="coerce"
    )

    chart_data = []

    for row_index, value in values.items():
        if pd.notna(value):
            chart_data.append((row_index, float(value)))

        if len(chart_data) >= CHART_MAX_ROWS:
            break

    if not chart_data:
        cv2.putText(
            frame,
            "No numeric values available.",
            (panel_x + 10, panel_y + 120),
            FONT,
            BODY_FONT_SCALE,
            WARNING_COLOR,
            1
        )
        return

    chart_top = panel_y + 115
    chart_bottom = panel_y + CHART_HEIGHT - 30

    label_width = 65
    value_width = 65

    bar_area_left = panel_x + label_width
    bar_area_right = panel_x + CHART_WIDTH - value_width

    bar_area_width = bar_area_right - bar_area_left

    row_height = (
        chart_bottom - chart_top
    ) // CHART_MAX_ROWS

    max_absolute_value = max(
        abs(value) for _, value in chart_data
    )

    if max_absolute_value == 0:
        max_absolute_value = 1

    for index, (row_index, value) in enumerate(chart_data):
        y = chart_top + index * row_height

        cv2.putText(
            frame,
            f"R{row_index + 1}",
            (panel_x + 8, y + 20),
            FONT,
            SMALL_FONT_SCALE,
            MUTED_TEXT,
            1
        )

        bar_width = int(
            abs(value) / max_absolute_value * bar_area_width
        )

        bar_width = max(1, bar_width)

        if value >= 0:
            bar_x1 = bar_area_left
            bar_x2 = min(
                bar_area_left + bar_width,
                bar_area_right
            )
        else:
            bar_x1 = max(
                bar_area_right - bar_width,
                bar_area_left
            )
            bar_x2 = bar_area_right

        cv2.rectangle(
            frame,
            (bar_x1, y + 5),
            (bar_x2, y + 25),
            BAR_COLOR,
            -1
        )

        cv2.putText(
            frame,
            f"{value:.2f}",
            (bar_area_right + 7, y + 20),
            FONT,
            SMALL_FONT_SCALE,
            TEXT_COLOR,
            1
        )

        cv2.line(
            frame,
            (panel_x + 5, y + row_height - 2),
            (panel_x + CHART_WIDTH - 5, y + row_height - 2),
            GRID_COLOR,
            1
        )


# ============================================================
# WEBCAM PREVIEW
# ============================================================

def draw_camera_preview(frame, webcam_frame, pinch_state):
    preview = cv2.resize(
        webcam_frame,
        (PREVIEW_WIDTH, PREVIEW_HEIGHT)
    )

    preview_x = (
        WINDOW_WIDTH - PREVIEW_WIDTH - PREVIEW_MARGIN
    )

    preview_y = (
        WINDOW_HEIGHT - PREVIEW_HEIGHT - PREVIEW_MARGIN
    )

    cv2.rectangle(
        frame,
        (
            preview_x - 3,
            preview_y - 3
        ),
        (
            preview_x + PREVIEW_WIDTH + 3,
            preview_y + PREVIEW_HEIGHT + 3
        ),
        ACCENT_COLOR if pinch_state else GRID_COLOR,
        2
    )

    frame[
        preview_y:preview_y + PREVIEW_HEIGHT,
        preview_x:preview_x + PREVIEW_WIDTH
    ] = preview

    cv2.putText(
        frame,
        "CAMERA PREVIEW",
        (preview_x, preview_y - 10),
        FONT,
        SMALL_FONT_SCALE,
        MUTED_TEXT,
        1
    )


# ============================================================
# GESTURE INTERACTION
# ============================================================

def find_header_at_position(cursor_x, cursor_y, header_regions):
    for region in header_regions:
        if (
            region["x1"] <= cursor_x <= region["x2"]
            and region["y1"] <= cursor_y <= region["y2"]
        ):
            return region["column"]

    return None


def is_over_table_body(cursor_x, cursor_y):
    table_body_top = TABLE_Y + TABLE_HEADER_HEIGHT

    table_body_bottom = (
        table_body_top
        + TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT
    )

    return (
        TABLE_X <= cursor_x <= TABLE_X + TABLE_WIDTH
        and table_body_top <= cursor_y <= table_body_bottom
    )


def update_scroll(
    cursor_y,
    previous_cursor_y,
    scroll_offset,
    dataframe_length,
    scroll_accumulator
):
    if previous_cursor_y is None:
        return scroll_offset, scroll_accumulator

    movement = previous_cursor_y - cursor_y
    scroll_accumulator += movement

    if abs(scroll_accumulator) >= SCROLL_PIXELS_PER_ROW:
        rows_to_scroll = int(
            scroll_accumulator / SCROLL_PIXELS_PER_ROW
        )

        rows_to_scroll = clamp(
            rows_to_scroll,
            -MAX_SCROLL_STEP,
            MAX_SCROLL_STEP
        )

        scroll_offset += int(rows_to_scroll)

        scroll_accumulator -= (
            rows_to_scroll * SCROLL_PIXELS_PER_ROW
        )

    maximum_offset = max(
        0,
        dataframe_length - TABLE_VISIBLE_ROWS
    )

    scroll_offset = int(
        clamp(scroll_offset, 0, maximum_offset)
    )

    return scroll_offset, scroll_accumulator


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gesture-controlled CSV dashboard"
    )

    parser.add_argument(
        "csv_path",
        help="Path to the CSV file to display"
    )

    args = parser.parse_args()

    # Load CSV
    try:
        dataframe = pd.read_csv(args.csv_path)
    except Exception as error:
        print(f"Could not load CSV file: {error}")
        return

    if dataframe.empty:
        print("The CSV file contains no data.")
        return

    if len(dataframe.columns) == 0:
        print("The CSV file contains no columns.")
        return

    print(f"Loaded: {args.csv_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    # Initial selected column: first numeric column, if available
    numeric_columns = get_numeric_columns(dataframe)
    selected_column = numeric_columns[0] if numeric_columns else None

    # Load MediaPipe model
    try:
        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS
        )

        detector = vision.HandLandmarker.create_from_options(options)

    except Exception as error:
        print(f"Could not load hand tracking model: {error}")
        print(f"Check model location: {MODEL_PATH}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Gesture state
    pinch_state = False
    previous_pinch_state = False
    pinch_start_region = None

    midpoint_history = deque(maxlen=SMOOTHING_WINDOW)

    previous_cursor = None
    scroll_accumulator = 0
    scroll_offset = 0

    previous_time = time.time()
    timestamp_ms = 0

    print("Dashboard started.")
    print("Pinch a column header to select it.")
    print("Pinch and drag vertically from the table body to scroll.")
    print("Drag upward to see later rows; drag downward to return.")
    print("Press Q to quit.")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while cap.isOpened():
            success, camera_frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            camera_frame = cv2.resize(
                camera_frame,
                (960, 540)
            )

            # Create exactly one dashboard canvas per frame.
            dashboard = np.zeros(
                (WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                dtype=np.uint8
            )

            # MediaPipe receives the unmirrored image.
            rgb_frame = cv2.cvtColor(
                camera_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = max(
                timestamp_ms + 1,
                int(time.monotonic() * 1000)
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Mirrored camera preview
            webcam_frame = cv2.flip(camera_frame, 1)

            draw_dashboard_background(dashboard)

            # Draw table and get clickable header regions.
            header_regions = draw_table(
                dashboard,
                dataframe,
                selected_column,
                scroll_offset
            )

            # Draw selected column chart.
            draw_chart(
                dashboard,
                dataframe,
                selected_column
            )

            cursor_x = None
            cursor_y = None

            # =================================================
            # HAND PROCESSING
            # =================================================

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(webcam_frame, landmarks)

                pinch_ratio = calculate_pinch_ratio(landmarks)

                previous_pinch_state = pinch_state
                pinch_state = update_pinch_state(
                    pinch_ratio,
                    pinch_state
                )

                pinch_started = (
                    pinch_state and not previous_pinch_state
                )
                pinch_released = (
                    previous_pinch_state and not pinch_state
                )

                midpoint = calculate_midpoint(landmarks)
                smoothed = smooth_midpoint(
                    midpoint,
                    midpoint_history
                )

                # Convert to mirrored webcam coordinates.
                normalised_x = 1.0 - smoothed[0]
                normalised_y = smoothed[1]

                # Map the cursor to the full dashboard.
                cursor_x, cursor_y = map_webcam_to_dashboard(
                    normalised_x,
                    normalised_y
                )

                # Latch the region where this pinch began.
                if pinch_started:
                    header_column = find_header_at_position(
                        cursor_x,
                        cursor_y,
                        header_regions
                    )

                    if header_column is not None:
                        pinch_start_region = "header"
                        selected_column = header_column
                        print(f"Selected column: {selected_column}")

                    elif is_over_table_body(cursor_x, cursor_y):
                        pinch_start_region = "table_body"
                        previous_cursor = (cursor_x, cursor_y)
                        scroll_accumulator = 0

                    else:
                        pinch_start_region = "other"
                        previous_cursor = None
                        scroll_accumulator = 0

                # Scroll only when the pinch STARTED in the table body.
                # Keep the cursor over the table while dragging.
                if (
                    pinch_state
                    and pinch_start_region == "table_body"
                    and is_over_table_body(cursor_x, cursor_y)
                    and previous_cursor is not None
                ):
                    previous_cursor_y = previous_cursor[1]

                    scroll_offset, scroll_accumulator = update_scroll(
                        cursor_y,
                        previous_cursor_y,
                        scroll_offset,
                        len(dataframe),
                        scroll_accumulator
                    )

                if pinch_state and pinch_start_region == "table_body":
                    # Track movement while the table-body gesture is active.
                    previous_cursor = (cursor_x, cursor_y)

                if pinch_released:
                    pinch_start_region = None
                    previous_cursor = None
                    scroll_accumulator = 0

                cursor_color = (
                    PINCH_COLOR if pinch_state else CURSOR_COLOR
                )

                cv2.circle(
                    dashboard,
                    (cursor_x, cursor_y),
                    12,
                    cursor_color,
                    2
                )

                cv2.circle(
                    dashboard,
                    (cursor_x, cursor_y),
                    3,
                    cursor_color,
                    -1
                )

                cv2.putText(
                    dashboard,
                    f"PINCH: {'ON' if pinch_state else 'OFF'}",
                    (WINDOW_WIDTH - 220, 45),
                    FONT,
                    BODY_FONT_SCALE,
                    cursor_color,
                    1
                )

                cv2.putText(
                    dashboard,
                    f"Ratio: {pinch_ratio:.3f}",
                    (WINDOW_WIDTH - 220, 70),
                    FONT,
                    SMALL_FONT_SCALE,
                    MUTED_TEXT,
                    1
                )

            else:
                # No hand detected: reset active gesture state.
                pinch_state = False
                previous_pinch_state = False
                pinch_start_region = None
                previous_cursor = None
                midpoint_history.clear()
                scroll_accumulator = 0

            # =================================================
            # FPS
            # =================================================

            current_time = time.time()
            elapsed_time = current_time - previous_time

            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            previous_time = current_time

            cv2.putText(
                dashboard,
                f"FPS: {int(fps)}",
                (WINDOW_WIDTH - 220, 100),
                FONT,
                SMALL_FONT_SCALE,
                MUTED_TEXT,
                1
            )

            # =================================================
            # CAMERA PREVIEW
            # =================================================

            draw_camera_preview(
                dashboard,
                webcam_frame,
                pinch_state
            )

            cv2.imshow(WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        # Always release hardware and close MediaPipe resources,
        # including when an exception occurs in the dashboard loop.
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    print("Dashboard closed.")


if __name__ == "__main__":
    main()
