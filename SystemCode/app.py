import random
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
from frame_process import draw_counting_lines, exit_count, get_one_target, owl_text_detection, load_owl_model
from dotenv import load_dotenv
from utils import logger
import requests
from PIL import Image
import torch
import supervision as sv
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import scipy
from cnn_model import CNNModel
from torchvision import transforms
from PIL import Image

# Suppress all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(
    page_title="Street Owl",
    page_icon=":owl:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_headers():
    temp_text = """
    <p style="background-color: purple; border-radius: 8 px; text-align: center; color: white;">Exit Left</p>
    """
    st.session_state.left_exits_header.html(temp_text)
    temp_text = """
    <p style="background-color: purple; border-radius: 8 px; text-align: center; color: white;">Exit Right</p>
    """
    st.session_state.right_exits_header.html(temp_text)
    temp_text = """
    <div style="background-color: orange; border-radius: 8 px; text-align: center; color: white;">Human Detected</div>
    """
    st.session_state.detected_header.html(temp_text)

    temp_text = """
    <p style="background-color: blue; border-radius: 8 px; text-align: center; color: white;">FPS</p>
    """
    st.session_state.fps_header.html(temp_text)




def init_page_config():
    if "init_flag" not in st.session_state:


        # Check for GPU availability
        device_name = "CPU"
        if (torch.cuda.is_available()):
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device)

        # load model
        model = CNNModel(num_classes=3)
        model.load_state_dict(torch.load("best_density_model.pt"))
        model.to(device)
        st.session_state.densitymodel = model
        st.session_state.device_name = device_name
        st.session_state.device = device

        logger.info("=========debug init===========")
        load_dotenv()

        _, title_col = st.columns([1,6])
        #logo_col.image("Images/streetowl_logo.png")
        # title_col.title("Street Owl Monitoring")
        st.session_state.title_col = st.html("<h1>Street Owl Monitoring</h1>")
        
        main_left, main_right = st.columns([2,1])
        with main_left:
            with st.expander("Show/Hide Video Frame", expanded=True):
                frame_placeholder = st.empty()
            st.session_state.frame_placeholder = frame_placeholder
        
        with main_right:
            st.session_state.target_image_placeholder = st.empty()
            st.session_state.analyse_result_placeholder = st.empty()

        # Create placeholders for metrics
        left_col, right_col = st.columns([1, 1])
        with left_col:
            temp_row = st.columns([1.2,0.9,0.9])
            with temp_row[0]: 
                temp_text = """
                <div style="background-color: grey; border-radius: 8 px; text-align: center; color: white;">Waiting...</div>
                """
                st.session_state.density_placeholder =  st.html(temp_text)
                _, _,image_url = get_density_display(1) # default
                st.session_state.density_state = 1
                st.session_state.density_image = st.image(image_url, use_column_width=True)

            with temp_row[1]:
                st.session_state.detected_header = st.html("")
                st.session_state.detected_placeholder = st.empty()

            with temp_row[2]:
                st.session_state.fps_header = st.html("")
                st.session_state.fps_placeholder = st.empty()


            temp_row = st.columns([1,1])
            with temp_row[0]:
                st.session_state.left_exits_header = st.html("")
                st.session_state.left_exits_placeholder = st.empty()
            with temp_row[1]:
                st.session_state.right_exits_header = st.html("")
                st.session_state.right_exits_placeholder = st.empty()
            
        with right_col:
            st.session_state.livechart_data = pd.DataFrame(columns=['Detected'])
            st.session_state.livechart_placeholder = st.line_chart(st.session_state.livechart_data)
        
        model, processor = load_owl_model()
        st.session_state.owl_model = model
        st.session_state.owl_processor = processor
        #Create a placeholder for the video frame
        st.session_state.init_flag = True

def analyse():
    st.write(st.session_state.data)

def get_options():
    sidebar, tracking, scan = st.sidebar.tabs(["Sidebar", "Track", "Scan"])
    with sidebar:
        sidebar.text(f"Using device:\n{st.session_state.device_name}")
        # Sidebar for user input
        sidebar.header("Settings")
        model_choice = sidebar.selectbox(
            "Select YOLO model",
            ("streetowlbest.pt","streetowl-segbest.pt","yolov8n.pt", "yolov8l.pt", "yolov8x.pt", "yolov8n-obb.pt", "yolov8n-seg.pt", "yolov8l-seg.pt","custom_yolov8s.pt")
        )
        url = sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=DjdUEyjx8GM")
        confidence_threshold = sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.1, 0.05)
        frame_skip = sidebar.number_input("Frame Skip", 0, 10, 2)
        vid_quality = sidebar.selectbox(
            "Select Video Quality",
            ("360p", "480p", "720p", "1080p", "best")
        )
    with tracking:
        s = """Gender: str, MALE or FEMALE, with confidence score
Clothing: Dict
Age: str, KID or YOUNG OR MID-AGE or OLD
Note: more information about the target"""
        st.text_input("Target Desc Text", key="target_desc_text")
        st.number_input("Target ID", 0, key="target_id")
        st.text_area("Result Format Requirement", s, key="result_requirement", height=200)
        st.button("Track", on_click=get_one_target)
    
    with scan:
        st.text_input("Owl Detection Input", key="owl_text")
        st.button("Scan", on_click=owl_text_detection)
    logger.info("=========option========")
    return model_choice, url, confidence_threshold, frame_skip, vid_quality

def get_density_display(level):
    if level == 1:
        return "green", "Sparse","Images/owls_sparse.png"
    elif level == 2:
        return "orange", "Dense","Images/owls_dense.png"
    elif level == 3:
        return "red", "Crowded","Images/owls_crowded.png"
    
def update_placeholders(annotated_frame):
    fps_placeholder = st.session_state.fps_placeholder
    detected_placeholder = st.session_state.detected_placeholder
    left_exits_placeholder = st.session_state.left_exits_placeholder
    right_exits_placeholder = st.session_state.right_exits_placeholder
    livechart_placeholder = st.session_state.livechart_placeholder
    livechart_data = st.session_state.livechart_data
    frame_placeholder = st.session_state.frame_placeholder
    density_placeholder = st.session_state.density_placeholder
    density_image = st.session_state.density_image
    density_state = st.session_state.density_state

    title_col = st.session_state.title_col
    title_col.html("<h1>Street Owl Monitoring</h1>")
    render_headers()

    # existing density level output
    dense_level = st.session_state.current_density + 1
    human_count = st.session_state.num_objects
    # ensemble results
    if human_count <= 5 and dense_level >= 2:
        dense_level = 1
    elif human_count <= 10 and dense_level >= 2:
        dense_level = 2


    # Update density state change
    st.session_state.track_density_history.append(dense_level)
    density_history = st.session_state.track_density_history
    # check for at least 20 frames before changing density state
    if len(density_history) >= 20:
        values, counts = np.unique(st.session_state.track_density_history, return_counts=True)
        mode_value = values[np.argmax(counts)]

        # logger.info(f"density state changed from {density_state} to {dense_level}")
        st.session_state.density_state  = mode_value
        color, label,image_url = get_density_display(mode_value)

        density_markdown = f"""
        <p style="background-color: {color}; border-radius: 8 px; text-align: center; color: white;">
            Density : {label}
        </p>
        """
        density_placeholder.html(density_markdown)
        density_image.image(image_url, use_column_width=True)
        # reset the state once the density state is changed
        st.session_state.track_density_history = []
    
    fps_placeholder.html(f" <div width='100%' align='center' valign='center'><font size=14>{int(st.session_state.fps)} </font></div>")
    detected_placeholder.html(f" <div width='100%' align='center'><font size=14>{int(st.session_state.num_objects)} </font></div>")

    left_exits_placeholder.html(f" <div width='100%' align='center'><font size=14>{int(st.session_state.left_exit_count)} </font></div>")
    right_exits_placeholder.html(f" <div width='100%' align='center'><font size=14>{int(st.session_state.right_exit_count)} </font></div>")

    # Update live chart
    if livechart_placeholder is not None:
        livechart_data.loc[len(livechart_data)] = st.session_state.num_objects
        if len(livechart_data) > 120:
            livechart_data = livechart_data.tail(120).reset_index(drop=True)
        # livechart_data.columns = ['Detected']
        livechart_placeholder.line_chart(livechart_data, y_label='People Detected')

    # Display the annotated frame
    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)


def add_overlay(frame):
    track_results = st.session_state.track_results
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

# for density inference
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

def cv2_to_pil(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def preprocess_frame(frame):
    pil_image = cv2_to_pil(frame)  # Convert the frame to PIL image
    image = preprocess(pil_image)  # Apply the transformations
    image = image.unsqueeze(0)     # Add batch dimension
    image = image.to(st.session_state.device)  
    return image

def predict_frame(model, frame):
    image_tensor = preprocess_frame(frame)  # Preprocess the frame
    predictions = model.predict(image_tensor)  # Predict class
    return predictions

def main():
    init_page_config()
    model_choice, url, confidence_threshold, frame_skip, vid_quality = get_options()

    # Load the YOLOv8 model
    @st.cache_resource
    def load_model(model_path):
        model = YOLO(model_path).to(st.session_state.device)
        logger.info(model.device)
        return model
    

    model = load_model(model_choice)

    # Initialize video capture
    cap = get_cap(url, vid_quality)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.session_state.left_exit_count = 0
    st.session_state.right_exit_count = 0
    st.session_state.track_history = {}
    st.session_state.left_exited_ids = set()
    st.session_state.right_exited_ids = set()
    st.session_state.track_density_history = []

    # Define counting lines
    st.session_state.frame_width = frame_width 
    st.session_state.frame_height = frame_height
    st.session_state.left_line = int(frame_width * 0.2)
    st.session_state.right_line = int(frame_width * 0.8)
    st.session_state.process_flag = True
    
    density_model = st.session_state.densitymodel    


    frame_counter = 0
    prev_frame_time = time.time()

    while (cap.isOpened()) and st.session_state.process_flag:
        success, frame = cap.read()
        frame_counter += 1
        
        if frame_skip > 0 and frame_counter % frame_skip != 0:
            continue

        if success:
            track_results = model.track(frame, persist=True, classes=0, conf=confidence_threshold, tracker="bytetrack.yaml",verbose=False)
            st.session_state.current_frame = frame
            st.session_state.track_results = track_results

            predicted_density = predict_frame(density_model, frame)
            st.session_state.current_density = predicted_density.item()

            annotated_frame = frame
            # annotated_frame = frame.copy()
            #annotated_frame = track_results[0].plot() # info from yolo v8, optional, can comment off
            # Create a blank overlay for the semi-transparent masks
            annotated_frame = add_overlay(annotated_frame)
            annotated_frame = exit_count(annotated_frame)
            # Draw counting lines
            draw_counting_lines(annotated_frame, track_results)

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1.0 / ((new_frame_time - prev_frame_time)+0.01)
            prev_frame_time = new_frame_time
            st.session_state.fps = fps

            # Count detected objects
            st.session_state.num_objects = len(track_results[0].boxes) if track_results[0].boxes is not None else 0
            
            update_placeholders(annotated_frame)
            st.session_state.annotated_frame = annotated_frame

        if not success:
            st.write("End of video stream.")
            break

    cap.release()

if __name__ == "__main__":
    main()