import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np
import streamlink
import torch 
import pandas as pd
import warnings
from get_cap import get_cap
from frame_process import draw_counting_lines, exit_count, get_one_target

# Suppress all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def init_page_config():
    st.set_page_config(
        page_title="Street Owl",
        page_icon=":owl:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    logo_col, title_col = st.columns([1,6])
    #logo_col.image("Images/streetowl_logo.png")
    title_col.title("Street Owl Monitoring")

    # Check for GPU availability
    device_name = "CPU"
    if (torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    st.sidebar.text(f"Using device:\n{device_name}")
    # Sidebar for user input
    st.sidebar.header("Settings")

def analyse():
    st.write(st.session_state.data)

def get_options():
    model_choice = st.sidebar.selectbox(
        "Select YOLO model",
        ("streetowlbest.pt","streetowl-segbest.pt","yolov8n.pt", "yolov8l.pt", "yolov8x.pt", "yolov8n-obb.pt", "yolov8n-seg.pt", "yolov8l-seg.pt","custom_yolov8s.pt")
    )
    url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=DjdUEyjx8GM")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.1, 0.05)
    frame_skip = st.sidebar.number_input("Frame Skip", 0, 10, 2)
    st.button("Test", type="primary", on_click=get_one_target)
    return model_choice, url, confidence_threshold, frame_skip


def update_placeholders(placeholders, data, annotated_frame):
    fps_placeholder = placeholders["fps_placeholder"]
    detected_placeholder = placeholders["fps_placeholder"]
    left_exits_placeholder = placeholders["left_exits_placeholder"]
    right_exits_placeholder = placeholders["right_exits_placeholder"]
    livechart_placeholder = placeholders["livechart_placeholder"]
    livechart_data = placeholders["livechart_data"]
    frame_placeholder = placeholders["frame_placeholder"]

    fps_placeholder.text(f"FPS: {int(data["fps"])}")
    detected_placeholder.text(f"Detected Objects: {data["num_objects"]}")
    left_exits_placeholder.text(f"Left Exits: {data["left_exit_count"]}")
    right_exits_placeholder.text(f"Right Exits: {data["right_exit_count"]}")

    # Update live chart
    if livechart_placeholder is not None:
        livechart_data.loc[len(livechart_data)] = data["num_objects"]
        if len(livechart_data) > 120:
            livechart_data = livechart_data.tail(120).reset_index(drop=True)
        # livechart_data.columns = ['Detected']
        livechart_placeholder.line_chart(livechart_data, y_label='People Detected')

    # Display the annotated frame
    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)   


def add_overlay(frame, track_results):
    overlay = np.zeros_like(frame, dtype=np.uint8)
    if track_results[0].masks is not None:
        for mask in track_results[0].masks.xy:
            # Convert the polygon to a format suitable for cv2.fillPoly
            polygon = np.array(mask, dtype=np.int32)
            
            # Fill the polygon on the overlay
            cv2.fillPoly(overlay, [polygon], color=(255, 0, 255))  # Mask color
    
    # Blend the overlay with the original frame
    alpha = 0.3  # Adjust this value to change the transparency (0.0 - 1.0)
    annotated_frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    return annotated_frame

def main():
    init_page_config()
    model_choice, url, confidence_threshold, frame_skip = get_options()

    # Load the YOLOv8 model
    @st.cache_resource
    def load_model(model_path):
        return YOLO(model_path)

    model = load_model(model_choice)
    vid_quality = st.sidebar.selectbox(
        "Select Video Quality",
        ("360p", "480p", "720p", "1080p", "best")
    )


    # Initialize video capture
    cap = get_cap(url, vid_quality)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Create a placeholder for the video frame
    with st.expander("Show/Hide Video Frame", expanded=True):
        frame_placeholder = st.empty()



    # Register data
    data = {}
    data["left_exit_count"] = 0
    data["right_exit_count"] = 0
    data["track_history"] = {}
    data["left_exited_ids"] = set()
    data["right_exited_ids"] = set()

    # Define counting lines
    data["frame_width"] = frame_width 
    data["frame_height"] = frame_height
    data["left_line"] = int(frame_width * 0.2)
    data["right_line"] = int(frame_width * 0.8)
    st.session_state.data = data


    # Create placeholders for metrics
    placeholders = {}
    left_col, right_col = st.columns([1, 1])
    with left_col:
        placeholders["fps_placeholder"] = st.empty()
        placeholders["detected_placeholder"] = st.empty()
        placeholders["left_exits_placeholder"] = st.empty()
        placeholders["right_exits_placeholder"] = st.empty()

    with right_col:
        placeholders["livechart_data"] = pd.DataFrame(columns=['Detected'])
        placeholders["livechart_placeholder"] = st.line_chart(placeholders["livechart_data"])
    placeholders["frame_placeholder"] = frame_placeholder
        


    frame_counter = 0
    prev_frame_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        frame_counter += 1
        
        if frame_skip > 0 and frame_counter % frame_skip != 0:
            continue

        if success:
            track_results = model.track(frame, persist=True, classes=0, conf=confidence_threshold, tracker="bytetrack.yaml",verbose=False)
            st.session_state.current_frame = frame
            st.session_state.track_results = track_results

            annotated_frame = frame
            # annotated_frame = frame.copy()
            #annotated_frame = track_results[0].plot() # info from yolo v8, optional, can comment off
            # Create a blank overlay for the semi-transparent masks
            annotated_frame = add_overlay(annotated_frame, track_results=track_results)
            annotated_frame, data = exit_count(annotated_frame, track_results, data)
            # Draw counting lines
            draw_counting_lines(annotated_frame, track_results, data)

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1.0 / ((new_frame_time - prev_frame_time)+0.01)
            prev_frame_time = new_frame_time
            data["fps"] = fps

            # Count detected objects
            data["num_objects"] = len(track_results[0].boxes) if track_results[0].boxes is not None else 0
            
            update_placeholders(placeholders, data, annotated_frame)

        if not success:
            st.write("End of video stream.")
            break

    cap.release()

if __name__ == "__main__":
    main()