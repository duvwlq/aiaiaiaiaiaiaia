import os
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from functools import lru_cache


def validate_files(files):
    """여러 파일 존재 여부를 한 번에 검증."""
    for file_path, file_description in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_description} 파일을 찾을 수 없습니다: {file_path}")


@lru_cache(maxsize=1)
def load_yolo_model(model_path):
    """YOLO 모델을 로드 및 캐싱."""
    return YOLO(model_path)


def predict_and_visualize(model_path, test_image_path, output_path="result_prediction.jpg", show_result=False):
    """모델로 이미지를 예측하고 시각화."""
    validate_files([(model_path, "모델"), (test_image_path, "테스트 이미지")])

    # 모델 로드 (캐싱된 모델 사용)
    model = load_yolo_model(model_path)

    # PIL로 이미지 로드
    image = Image.open(test_image_path).convert("RGB")
    image = np.array(image)

    # 예측 수행
    print("모델 추론 중...")
    results = model.predict(source=test_image_path, conf=0.25, device="cuda")  # GPU 사용

    # 시각화 및 저장
    result_image = results[0].plot()
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"결과 이미지가 저장되었습니다: {output_path}")

    # 화면 시각화
    if show_result:
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.axis("off")
        plt.title("Prediction Result")
        plt.show()

    # 결과 정보 출력
    for i, box_data in enumerate(results[0].boxes):
        cls = int(box_data.cls[0])
        conf = float(box_data.conf[0])
        label = model.names[cls]
        print(f"객체 {i + 1}: {label} (신뢰도: {conf:.2f})")


if __name__ == "__main__":
    model_path = r"models/best.pt"
    test_image_path = r"image_모음/test_image/20250320_204956.jpg"
    output_image_path = r"image_모음/결과_image/result.png"

    try:
        predict_and_visualize(model_path, test_image_path, output_path=output_image_path, show_result=True)
    except Exception as e:
        print(f"오류 발생: {e}")
