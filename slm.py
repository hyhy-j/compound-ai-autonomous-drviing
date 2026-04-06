from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델 로드
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")
print("모델 로드 중... (처음엔 다운로드로 시간 걸려요)")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def analyze_scene(detections):
    # 탐지 결과를 텍스트로 변환
    if not detections:
        return "탐지된 객체 없음"
    
    detection_text = ""
    for d in detections:
        detection_text += f"- {d['direction']} {d['distance']}: {d['class']} (신뢰도: {d['confidence']})\n"
    
    prompt = f"""당신은 자율주행 보조 AI입니다. 아래 예시처럼 탐지된 객체를 보고 정확히 3문장으로 주행 상황을 분석하세요.

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
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# 테스트
test_detections = [
    {"class": "car", "confidence": 0.86, "direction": "중앙", "distance": "근거리"},
    {"class": "person", "confidence": 0.75, "direction": "좌측", "distance": "근거리"},
    {"class": "traffic light", "confidence": 0.90, "direction": "중앙", "distance": "중거리"},
]

print("\n=== 맥락 추론 결과 ===")
result = analyze_scene(test_detections)
print(result)