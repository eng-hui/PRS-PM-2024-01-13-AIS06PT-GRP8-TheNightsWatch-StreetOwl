import streamlink
import cv2

def get_cap(url, vid_quality):
    streams = streamlink.streams(url)
    video_url = streams[vid_quality].url
    cap = cv2.VideoCapture(video_url)
    return cap