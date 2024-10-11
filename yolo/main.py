from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np


def callback(image_slice: np.array):
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

if __name__ == "__main__" : 

    source_video_path = "20241007.mov"
    target_video_path = "output.mp4"
    confidence_threshold = 0.3
    iou_threshold = 0.7

    model = YOLO("yolov8m.pt")

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            # slicer = sv.InferenceSlicer(callback=callback)
            # detections = slicer(image=frame)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)
    
