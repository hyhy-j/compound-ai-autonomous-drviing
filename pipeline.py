import sys
sys.path.append("/home/hyeonjeong/autonomous_project/Depth-Anything-V2")

from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import cv2
import os
import json
import numpy as np
import time

from kafka import KafkaProducer
import json

# 경로 설정
IMAGE_DIR = "/home/hyeonjeong/autonomous_project/100k/train"
OUTPUT_DIR = "/home/hyeonjeong/autonomous_project/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# YOLOv11 로드
print("YOLOv11 로드 중...")
yolo_model = YOLO("yolo11n.pt")

# Depth Anything V2 로드
print("Depth Anything V2 로드 중...")
depth_model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load(
    "/home/hyeonjeong/autonomous_project/Depth-Anything-V2/depth_anything_v2_vits.pth",
    map_location=device
))
depth_model = depth_model.to(device).eval()

# SLM 로드
print("SLM 로드 중...")
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
slm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)

# Kafka Producer 설정
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
)

def get_spatial_info(bbox, depth_map, img_width, img_height):
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    depth_region = depth_map[y1:y2, x1:x2]
    avg_depth = float(np.mean(depth_region)) if depth_region.size > 0 else 0

    if cx < img_width / 3:
        direction = "좌측"
    elif cx < img_width * 2 / 3:
        direction = "중앙"
    else:
        direction = "우측"

    if avg_depth > 0.7:
        distance = "근거리"
    elif avg_depth > 0.4:
        distance = "중거리"
    else:
        distance = "원거리"

    return direction, distance

def analyze_scene(detections):
    if not detections:
        return "탐지된 객체가 없습니다. 정상 주행하세요.", 0.0

    detection_text = ""
    for d in detections:
        detection_text += f"- {d['direction']} {d['distance']}: {d['class']} (신뢰도: {d['confidence']})\n"

    prompt = f"""자율주행 AI입니다. 반드시 한국어 3문장으로만 답하세요. 다른 내용 출력 금지.

예시1:
탐지된 객체:
- 중앙 근거리: 차량 (신뢰도: 0.85)
- 좌측 근거리: 보행자 (신뢰도: 0.75)
분석: 전방 차량과 좌측 보행자가 감지되었습니다. 즉시 감속하고 보행자가 통과할 때까지 대기하세요. 안전 거리를 유지하며 서행하세요.

예시2:
탐지된 객체:
- 중앙 원거리: 신호등 (신뢰도: 0.90)
- 우측 중거리: 차량 (신뢰도: 0.80)
분석: 전방 신호등과 우측 차량이 감지되었습니다. 신호등 상태를 확인하며 속도를 줄이세요. 우측 차량과의 안전 거리를 유지하세요.

예시3:
탐지된 객체:
- 좌측 근거리: 자전거 (신뢰도: 0.70)
- 중앙 중거리: 차량 (신뢰도: 0.65)
분석: 좌측 근거리에 자전거가 감지되었습니다. 자전거와의 충돌 위험이 있으므로 즉시 감속하세요. 전방 차량과의 거리를 유지하며 주의하여 주행하세요.

이제 아래 상황을 분석하세요:
탐지된 객체:
{detection_text}
분석:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time = time.time()

    with torch.no_grad():
        outputs = slm_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True
        )

    slm_time = time.time() - start_time
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 한국어 문장만 추출
    sentences = []
    for sent in response.replace('\n', ' ').split('.'):
        sent = ''.join(c for c in sent if c not in '*#_[]()').strip()
        if sent and any('\uac00' <= c <= '\ud7a3' for c in sent):
            sentences.append(sent)
        if len(sentences) >= 3:
            break

    if sentences:
        clean_response = '. '.join(sentences) + '.'
    else:
        clean_response = "주변 상황을 주의하며 안전하게 주행하세요."

    return clean_response, slm_time

# GPU 워밍업
print("GPU 워밍업 중...")
dummy_img = cv2.imread(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0]))
yolo_model(dummy_img, verbose=False)
depth_model.infer_image(cv2.cvtColor(dummy_img, cv2.COLOR_BGR2RGB))
print("워밍업 완료!")

# 이미지 테스트
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])[:100]
results_list = []

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # YOLO 탐지
    start_time = time.time()
    yolo_results = yolo_model(img_path, verbose=False)
    yolo_time = time.time() - start_time

    # Depth 추정
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_map = depth_model.infer_image(img_rgb)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    detections = []
    for result in yolo_results:
        for box in result.boxes:
            direction, distance = get_spatial_info(box.xyxy[0].tolist(), depth_map, w, h)
            detection = {
                "class": yolo_model.names[int(box.cls)],
                "confidence": round(float(box.conf), 2),
                "direction": direction,
                "distance": distance
            }
            detections.append(detection)

    # SLM 맥락추론
    analysis, slm_time = analyze_scene(detections)

    result = {
        "image": img_file,
        "detections": detections,
        "analysis": analysis,
        "yolo_time": round(yolo_time, 3),
        "slm_time": round(slm_time, 3),
        "total_time": round(yolo_time + slm_time, 3)
    }
    results_list.append(result)

   # Kafka로 결과 전송
    try:
        producer.send('autonomous-result', value={
            "image": img_file,
            "analysis": analysis,
            "yoloTime": round(yolo_time, 3),
            "slmTime": round(slm_time, 3),
            "totalTime": round(yolo_time + slm_time, 3),
            "detectionCount": len(detections)
        })
        producer.flush()
        print(f"Kafka 전송 완료: {img_file}")
    except Exception as e:
        print(f"Kafka 전송 실패 (계속 진행): {e}")
        
    print(f"\n{'='*50}")
    print(f"이미지: {img_file}")
    print(f"탐지 객체 수: {len(detections)}개")
    print(f"YOLO 시간: {yolo_time:.3f}s")
    print(f"SLM 시간: {slm_time:.3f}s")
    print(f"전체 시간: {yolo_time + slm_time:.3f}s")
    print(f"맥락 추론: {analysis}")

# 결과 저장
with open(os.path.join(OUTPUT_DIR, "pipeline_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2, ensure_ascii=False)

print(f"\n완료! output/pipeline_results.json 확인해봐요")