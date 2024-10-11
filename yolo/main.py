from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse


def callback(image_slice: np.array):
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_video_path", type=str, required=True)
    parser.add_argument("-t", "--target_video_path", type=str, required=True)
    parser.add_argument("-c", "--confidence_threshold", type=float, default=0.3)
    parser.add_argument("-i", "--iou_threshold", type=float, default=0.7)
    args = parser.parse_args()

    model = YOLO("yolov8m.pt")

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    with sv.VideoSink(target_path=args.target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            results = model(
                frame, verbose=False, conf=args.confidence_threshold, iou=args.iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            sink.write_frame(frame=annotated_labeled_frame)
    
