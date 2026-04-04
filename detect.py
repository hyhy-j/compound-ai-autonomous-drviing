import sys
sys.path.append("/home/hyeonjeong/autonomous_project/Depth-Anything-V2")

from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import os
import json
import torch
import numpy as np

# 경로 설정
IMAGE_DIR = "/home/hyeonjeong/autonomous_project/100k/train"
OUTPUT_DIR = "/home/hyeonjeong/autonomous_project/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLOv11 모델 로드
yolo_model = YOLO("yolo11n.pt")

# Depth Anything V2 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load(
    "/home/hyeonjeong/autonomous_project/Depth-Anything-V2/depth_anything_v2_vits.pth",
    map_location=device
))
depth_model = depth_model.to(device).eval()

# 공간 좌표 추출 함수
def get_spatial_info(bbox, depth_map, img_width, img_height):
    x1, y1, x2, y2 = map(int, bbox)
    
    # 바운딩 박스 중심점
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # 깊이 추정 (바운딩 박스 내 평균)
    depth_region = depth_map[y1:y2, x1:x2]
    avg_depth = float(np.mean(depth_region)) if depth_region.size > 0 else 0
    
    # 방향 (좌/중/우)
    if cx < img_width / 3:
        direction = "좌측"
    elif cx < img_width * 2 / 3:
        direction = "중앙"
    else:
        direction = "우측"
    
    # 거리 (근/중/원)
    if avg_depth > 0.7:
        distance = "근거리"
    elif avg_depth > 0.4:
        distance = "중거리"
    else:
        distance = "원거리"
    
    return direction, distance, avg_depth

# 이미지 5장 테스트
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")][:5]
results_list = []

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # YOLO 탐지
    yolo_results = yolo_model(img_path, verbose=False)
    
    # Depth 추정
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_map = depth_model.infer_image(img_rgb)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    detections = []
    for result in yolo_results:
        for box in result.boxes:
            direction, distance, avg_depth = get_spatial_info(
                box.xyxy[0].tolist(), depth_map, w, h
            )
            detection = {
                "class": yolo_model.names[int(box.cls)],
                "confidence": round(float(box.conf), 2),
                "direction": direction,
                "distance": distance,
                "depth_value": round(avg_depth, 3)
            }
            detections.append(detection)
    
    results_list.append({
        "image": img_file,
        "detections": detections
    })
    
    print(f"\n{img_file}:")
    for d in detections:
        print(f"  {d['direction']} {d['distance']} - {d['class']} (신뢰도: {d['confidence']})")

# 결과 저장
with open(os.path.join(OUTPUT_DIR, "detection_results.json"), "w") as f:
    json.dump(results_list, f, indent=2, ensure_ascii=False)

print("\n완료! output/detection_results.json 확인해봐요")