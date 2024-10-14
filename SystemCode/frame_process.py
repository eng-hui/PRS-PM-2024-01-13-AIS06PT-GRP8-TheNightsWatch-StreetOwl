import cv2
import numpy as np
import streamlit as st
from utils import call_gpt

import numpy as np
import base64
from io import BytesIO
from PIL import Image
from utils import logger
import requests
import torch
import supervision as sv
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import scipy
import asyncio


# Load the Owlv2 model
@st.cache_resource
def load_owl_model():
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    return processor, model

def numpy_to_base64(image_np: np.ndarray) -> str:
    image_pil = Image.fromarray(image_np)
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def draw_counting_lines(annotated_frame, track_results):
    left_line = st.session_state.left_line
    right_line = st.session_state.right_line
    frame_height = st.session_state.frame_height

    cv2.line(annotated_frame, (left_line, 0), (left_line, frame_height), (255, 255, 0), 2)
    cv2.line(annotated_frame, (right_line, 0), (right_line, frame_height), (255, 255, 0), 2)    
    return annotated_frame


# Function to check if a person has crossed a line
def has_crossed_line(prev_pos, curr_pos, line_pos):
    return (prev_pos < line_pos and curr_pos >= line_pos) or (prev_pos > line_pos and curr_pos <= line_pos)


def desc_to_target_id(target_desc_text):
    annotated_frame = st.session_state.annotated_frame
    target_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    #st.image(target_image)
    img_str = numpy_to_base64(target_image)
    prompt = f"""extract the target_id from user's input
===user_input===
{target_desc_text}
===user_input end===
return in standard json format like:"""+\
"""
        {
            "target_id": int
        }
        """
    logger.info(prompt)
    messages = [
            {"role": "system", "content": "You are here to help analyse image from monitor."},
            {
                "role": "user",
                "content":[{
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{img_str}"
                    }}]
            },
            {
                "role": "user",
                "content": prompt
            }
    ]
    d = call_gpt(messages)
    logger.info(f"get target_id: {d.get('target_id')}")
    return d.get("target_id")


def owl_text_detection(text, frame):
    texts = [[text]]
    target_sizes = torch.Tensor([frame.shape[:2]])
    model, processor = load_owl_model()
    inputs = processor(text=texts, images=frame, return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)
    result = results[0]
    


def get_one_target():
    logger.info("===========start=========")
    target_track_id = st.session_state.target_id
    logger.info(st.session_state.target_id)

    if ("target_desc_text" in st.session_state) and (st.session_state.target_desc_text!=""):
        target_track_id  = desc_to_target_id(st.session_state.target_desc_text)
        st.session_state.target_track_id = target_track_id
    elif target_track_id is not None:
        st.session_state.target_track_id = target_track_id
    else:
        return 

    

    frame = st.session_state.current_frame
    track_results = st.session_state.track_results
    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
        for box, track_id in zip(track_results[0].boxes.xywh, track_results[0].boxes.id):
            if track_id == target_track_id:
                logger.info(f"{track_id} matched")
                box = track_results[0].boxes.xywh[0]
                x_center, y_center, w, h  = box
                # Calculate the top-left corner from the center (ensure the values are integers)
                x = int(x_center - w // 2)
                y = int(y_center - h // 2)

                # Ensure that the calculated coordinates are within the image bounds
                x = max(0, x)  # Make sure x is not less than 0
                y = max(0, y)  # Make sure y is not less than 0
                x_end = min(x + w, frame.shape[1])  # Ensure the width doesn't exceed image width
                y_end = min(y + h, frame.shape[0])  # Ensure the height doesn't exceed image height

                # get some more backgound infomation
                x = max(0, x*0.9) 
                y = max(0, y*0.9) 
                x_end = min(x_end*1.1, frame.shape[1]) 
                y_end = min(y_end*1.1, frame.shape[1]) 

                # Crop the target image from the frame using NumPy slicing
                target_image = frame[int(y):int(y_end), int(x):int(x_end)]
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                #st.image(target_image)
                st.session_state.target_image_placeholder.image(target_image)
                img_str = numpy_to_base64(target_image)
                prompt = """Analyse the above image like you are a super detective, return infomation from the person in the center of the image
        return in standard json format like:
        {
            "gender": {"value":str(MALE or FEMALE), "confidence":float}
            "clothing": Dict,
            "age": {"value":str(KID or YOUNG OR MID-AGE or OLD), "confidence":float},
            "note": text
        }
        """
                messages = [
                        {"role": "system", "content": "You are here to help analyse image from monitor."},
                        {
                            "role": "user",
                            "content":[{
                                "type": "image_url",
                                "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                                }}]
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                ]
                d = call_gpt(messages)
                st.session_state.analyse_result_placeholder.text(d)



def owl_full_image_detect(): 
    if ("target_desc_text" in st.session_state) and (st.session_state.target_desc_text!=""):
        text = st.session_state.target_desc_text
    else:
        return 

    frame = st.session_state.current_frame
    owl_text_detection(text)





def exit_count(annotated_frame):
    track_results = st.session_state.track_results
    left_line = st.session_state.left_line
    right_line = st.session_state.right_line
    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
        for box, track_id in zip(track_results[0].boxes.xywh, track_results[0].boxes.id):
            x, y, w, h = box
            track_id = int(track_id)
            center_x, center_y = int(x), int(y)

            if track_id not in st.session_state.track_history:
                st.session_state.track_history[track_id] = []
            st.session_state.track_history[track_id].append((center_x, center_y))
            st.session_state.track_history[track_id] = st.session_state.track_history[track_id][-30:]

            if len(st.session_state.track_history[track_id]) > 1:
                prev_x = np.mean([pos[0] for pos in st.session_state.track_history[track_id][:-10]])
                curr_x = np.mean([pos[0] for pos in st.session_state.track_history[track_id][-10:]])

                if has_crossed_line(prev_x, curr_x, left_line) and track_id not in st.session_state.left_exited_ids:
                    st.session_state.left_exit_count += 1
                    st.session_state.left_exited_ids.add(track_id)
                elif has_crossed_line(prev_x, curr_x, right_line) and track_id not in st.session_state.right_exited_ids:
                    st.session_state.right_exit_count += 1
                    st.session_state.right_exited_ids.add(track_id)

            if len(st.session_state.track_history[track_id]) > 1:
                cv2.polylines(annotated_frame, [np.array(st.session_state.track_history[track_id], dtype=np.int32)], False, (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_frame
            