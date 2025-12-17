import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import math
import os

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Vehicle Speed Detection & Tracking System",
    layout="wide"
)
st.title("🚗 Vehicle Speed Detection & Tracking System")
st.caption("Plain OpenCV | No Haar | No ML | Streamlit Cloud Ready")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
video_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi"])

PIXELS_PER_METER = 50     # calibration constant
DISPLAY_FPS = 15          # smooth playback control

# --------------------------------------------------
# BACKGROUND SUBTRACTOR
# --------------------------------------------------
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=40,
    detectShadows=False
)

# --------------------------------------------------
# TRACKING STORAGE
# --------------------------------------------------
vehicle_id = 0
vehicles = {}  # id -> (cx, cy, timestamp)

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------
if video_file:
    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(video_file.read())
        input_path = temp_input.name

    cap = cv2.VideoCapture(input_path)

    # Output video setup
    output_path = os.path.join(
        tempfile.gettempdir(), "vehicle_speed_output.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, DISPLAY_FPS, (640, 360))

    frame_placeholder = st.empty()

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        fg_mask = bg_subtractor.apply(gray)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(
            fg_mask, 200, 255, cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_vehicles = {}

        for cnt in contours:
            if cv2.contourArea(cnt) < 600:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            matched_id = None
            for vid, (px, py, pt) in vehicles.items():
                if math.hypot(cx - px, cy - py) < 50:
                    matched_id = vid
                    break

            if matched_id is None:
                vehicle_id += 1
                current_vehicles[vehicle_id] = (cx, cy, time.time())
            else:
                px, py, pt = vehicles[matched_id]
                dt = time.time() - pt
                dist_pixels = math.hypot(cx - px, cy - py)

                speed = 0
                if dt > 0:
                    speed = (dist_pixels / PIXELS_PER_METER) / dt * 3.6

                current_vehicles[matched_id] = (cx, cy, time.time())

                cv2.putText(
                    frame,
                    f"{int(speed)} km/h",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

        vehicles = current_vehicles

        # Save frame to output video
        out.write(frame)

        # Display smoothly in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Frame pacing
        elapsed = time.time() - start_time
        delay = max(0, (1 / DISPLAY_FPS) - elapsed)
        time.sleep(delay)

    cap.release()
    out.release()

    st.success("✅ Video processing completed")

    # --------------------------------------------------
    # DOWNLOAD BUTTON
    # --------------------------------------------------
    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="vehicle_speed_output.mp4",
            mime="video/mp4"
        )
