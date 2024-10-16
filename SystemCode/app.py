import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np
import streamlink
import torch 

def main():
    st.title("NUS ISS Traffic Monitoring")
    # Check for GPU availability
    device_name = "CPU"
    if (torch.cuda.is_available()):
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    st.sidebar.text(f"Using device:\n{device_name}")
    
    # Sidebar for user input
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox(
        "Select YOLO model",
        ("office_worker_best_train44.pt","placeholder.pt", 
         )
    )
    url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=DjdUEyjx8GM")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.1, 0.05)
    frame_skip = st.sidebar.number_input("Frame Skip", 0, 10, 2)

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
    if st.sidebar.button("Start Tracking"):
        streams = streamlink.streams(url)
        video_url = streams[vid_quality].url
        cap = cv2.VideoCapture(video_url)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define counting lines
        left_line = int(frame_width * 0.2)
        right_line = int(frame_width * 0.8)

        # Initialize counters and tracking variables
        left_exit_count = 0
        right_exit_count = 0
        track_history = {}
        left_exited_ids = set()
        right_exited_ids = set()

        # Create a placeholder for the video frame
        frame_placeholder = st.empty()
        
        # Create placeholders for metrics
        fps_placeholder = st.empty()
        detected_placeholder = st.empty()
        left_exits_placeholder = st.empty()
        right_exits_placeholder = st.empty()

        # Function to check if a person has crossed a line
        def has_crossed_line(prev_pos, curr_pos, line_pos):
            return (prev_pos < line_pos and curr_pos >= line_pos) or (prev_pos > line_pos and curr_pos <= line_pos)

        frame_counter = 0
        prev_frame_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            frame_counter += 1
            
            if frame_skip > 0 and frame_counter % frame_skip != 0:
                continue

            if success:
                # results = model.track(frame, persist=True, classes=0, conf=confidence_threshold, tracker="bytetrack.yaml",verbose=False)
                results = model.track(frame, persist=True, classes=[0,1], conf=confidence_threshold, tracker="bytetrack.yaml",verbose=False)

                annotated_frame = frame.copy()
                #annotated_frame = results[0].plot() # info from yolo v8, optional, can comment off
                # Create a blank overlay for the semi-transparent masks
                overlay = np.zeros_like(frame, dtype=np.uint8)
                if results[0].masks is not None:
                    for mask in results[0].masks.xy:
                        # Convert the polygon to a format suitable for cv2.fillPoly
                        polygon = np.array(mask, dtype=np.int32)
                        
                        # Fill the polygon on the overlay
                        cv2.fillPoly(overlay, [polygon], color=(255, 0, 255))  # Mask color
                
                # Blend the overlay with the original frame
                alpha = 0.3  # Adjust this value to change the transparency (0.0 - 1.0)
                annotated_frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)
                
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box, track_id in zip(results[0].boxes.xywh, results[0].boxes.id):
                        x, y, w, h = box
                        track_id = int(track_id)
                        center_x, center_y = int(x), int(y)

                        if track_id not in track_history:
                            track_history[track_id] = []
                        track_history[track_id].append((center_x, center_y))
                        track_history[track_id] = track_history[track_id][-30:]

                        if len(track_history[track_id]) > 1:
                            prev_x = np.mean([pos[0] for pos in track_history[track_id][:-10]])
                            curr_x = np.mean([pos[0] for pos in track_history[track_id][-10:]])

                            if has_crossed_line(prev_x, curr_x, left_line) and track_id not in left_exited_ids:
                                left_exit_count += 1
                                left_exited_ids.add(track_id)
                            elif has_crossed_line(prev_x, curr_x, right_line) and track_id not in right_exited_ids:
                                right_exit_count += 1
                                right_exited_ids.add(track_id)

                        if len(track_history[track_id]) > 1:
                            cv2.polylines(annotated_frame, [np.array(track_history[track_id], dtype=np.int32)], False, (0, 255, 0), 2)

                        cv2.putText(annotated_frame, f"ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        
  
                # Draw counting lines
                cv2.line(annotated_frame, (left_line, 0), (left_line, frame_height), (255, 255, 0), 2)
                cv2.line(annotated_frame, (right_line, 0), (right_line, frame_height), (255, 255, 0), 2)

                # Calculate FPS
                new_frame_time = time.time()
                fps = 1.0 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # Count detected objects
                num_objects = len(results[0].boxes) if results[0].boxes is not None else 0

                # Update metrics
                fps_placeholder.text(f"FPS: {int(fps)}")
                detected_placeholder.text(f"Detected Objects: {num_objects}")
                left_exits_placeholder.text(f"Left Exits: {left_exit_count}")
                right_exits_placeholder.text(f"Right Exits: {right_exit_count}")

                # Display the annotated frame
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

            if not success:
                st.write("End of video stream.")
                break

        cap.release()

if __name__ == "__main__":
    main()