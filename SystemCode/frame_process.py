import cv2
import numpy as np

def draw_counting_lines(annotated_frame, track_results, data):
    left_line = data.get("left_line")
    right_line = data.get("right_line")
    frame_height = data.get("frame_height")

    cv2.line(annotated_frame, (left_line, 0), (left_line, frame_height), (255, 255, 0), 2)
    cv2.line(annotated_frame, (right_line, 0), (right_line, frame_height), (255, 255, 0), 2)    
    return annotated_frame


# Function to check if a person has crossed a line
def has_crossed_line(prev_pos, curr_pos, line_pos):
    return (prev_pos < line_pos and curr_pos >= line_pos) or (prev_pos > line_pos and curr_pos <= line_pos)


def exit_count(annotated_frame, track_results, data):
    left_line = data.get("left_line")
    right_line = data.get("right_line")
    if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
        for box, track_id in zip(track_results[0].boxes.xywh, track_results[0].boxes.id):
            x, y, w, h = box
            track_id = int(track_id)
            center_x, center_y = int(x), int(y)

            if track_id not in data["track_history"]:
                data["track_history"][track_id] = []
            data["track_history"][track_id].append((center_x, center_y))
            data["track_history"][track_id] = data["track_history"][track_id][-30:]

            if len(data["track_history"][track_id]) > 1:
                prev_x = np.mean([pos[0] for pos in data["track_history"][track_id][:-10]])
                curr_x = np.mean([pos[0] for pos in data["track_history"][track_id][-10:]])

                if has_crossed_line(prev_x, curr_x, left_line) and track_id not in data["left_exited_ids"]:
                    data["left_exit_count"] += 1
                    data["left_exited_ids"].add(track_id)
                elif has_crossed_line(prev_x, curr_x, right_line) and track_id not in data["right_exited_ids"]:
                    data["right_exit_count"] += 1
                    data["right_exited_ids"].add(track_id)

            if len(data["track_history"][track_id]) > 1:
                cv2.polylines(annotated_frame, [np.array(data["track_history"][track_id], dtype=np.int32)], False, (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_frame, data
            