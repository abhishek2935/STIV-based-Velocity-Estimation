import pyorc

def load_video_frame(video_path, frame_index=0):
    video = pyorc.Video(video_path, start_frame=frame_index, end_frame=frame_index + 1)
    frame = video.get_frame(frame_index, method="rgb")
    return frame
