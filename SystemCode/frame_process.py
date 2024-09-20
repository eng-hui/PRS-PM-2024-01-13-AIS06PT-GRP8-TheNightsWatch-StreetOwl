import cv2
import numpy as np
import streamlit as st
from utils import call_gpt

import numpy as np
import base64
from io import BytesIO
from PIL import Image

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


def get_one_target():
    frame = st.session_state.current_frame
    track_results = st.session_state.track_results
    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
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

        # Crop the target image from the frame using NumPy slicing
        target_image = frame[int(y):int(y_end), int(x):int(x_end)]
        st.image(target_image)
        img_str = numpy_to_base64(target_image)
        prompt = """Analyse the above image like you are a super detective, return infomation from the person in the center of the image
return in standard json format like:
{
    "gender": str,
    "clothing": Dict,
    "age": str,
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
        st.write(d)



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
            