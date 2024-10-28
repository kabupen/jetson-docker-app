import torch
import argparse
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import numpy as np


def callback(image_slice: np.array):
    results = model.infer(image_slice)[0]
    return sv.Detections.from_inference(results)

keypoint_pairs = [
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
    (5, 7),  # 右肩→右肘
    (7, 9),  # 右肘→右手首
    (6, 8),  # 左肩→左肘
    (8, 10), # 左肘→左手首
    (11, 13), # 右腰→右膝
    (13, 15), # 右膝→右足首
    (12, 14), # 左腰→左膝
    (14, 16), # 左膝→左足首
    (5, 6),  # 右肩→左肩
    (11, 12), # 右腰→左腰
    (5, 11),
    (6, 12),
]

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_video_path", type=str, required=True)
    parser.add_argument("-t", "--target_video_path", type=str, required=True)
    parser.add_argument("-c", "--confidence_threshold", type=float, default=0.3)
    parser.add_argument("-i", "--iou_threshold", type=float, default=0.7)
    args = parser.parse_args()

    model = YOLO("yolov8n-pose.pt").to('cuda')

    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)

    with sv.VideoSink(target_path=args.target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # frame = torch.tensor(frame.transpose(2,0,1)).to("cuda").unsqueeze(0)
            # results = model(frame)[0]
            # results = model.predict(frame, device="cuda")[0]
            results = model(frame)[0]
            keypoints = results[0].keypoints.xy 
            keypoints = keypoints[0]

            # キーポイントを描画
            for x, y in keypoints:
                if x == 0. or y == 0. : continue
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 緑色で描画

            # 関節間の線を描画
            for pair in keypoint_pairs:
                pt1 = keypoints[pair[0]]
                pt2 = keypoints[pair[1]]
                if pt1[0] == 0. or pt2[0] == 0. : continue
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)  # 青色で線を描画

            sink.write_frame(frame=frame)
    
