import streamlit as st
import cv2
import time
import numpy as np
import streamlink
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define the neural network architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet18(pretrained=False)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

def main():
    st.title("NUS ISS Traffic Monitoring")
    # Check for GPU availability
    device_name = "CPU"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
    st.sidebar.text(f"Using device:\n{device_name}")
    
    # Sidebar for user input
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox(
        "Select YOLO model",
        ("simple_cnn_model.pt","placeholder.pt", 
         )
    )
    url = st.sidebar.text_input("YouTube URL", "https://www.youtube.com/watch?v=DjdUEyjx8GM")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.1, 0.05)
    frame_skip = st.sidebar.number_input("Frame Skip", 0, 10, 2)

    # Load the classification model
    @st.cache_resource
    def load_model(model_path):
        model = SimpleCNN(num_classes=2)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

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

        # Create a placeholder for the video frame
        frame_placeholder = st.empty()
        
        # Create placeholders for metrics
        fps_placeholder = st.empty()
        classification_placeholder = st.empty()

        frame_counter = 0
        prev_frame_time = time.time()

        # Define the transformation for the input image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        while cap.isOpened():
            success, frame = cap.read()
            frame_counter += 1
            
            if frame_skip > 0 and frame_counter % frame_skip != 0:
                continue

            if success:
                # Preprocess the frame
                input_tensor = transform(frame).unsqueeze(0).to(device)

                # Perform classification
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    classification_result = "Crowded" if predicted.item() == 1 else "Uncrowded"

                # Calculate FPS
                new_frame_time = time.time()
                fps = 1.0 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # Update metrics
                fps_placeholder.text(f"FPS: {int(fps)}")
                classification_placeholder.text(f"Classification: {classification_result}")

                # Display the frame
                frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            if not success:
                st.write("End of video stream.")
                break

        cap.release()

if __name__ == "__main__":
    main()