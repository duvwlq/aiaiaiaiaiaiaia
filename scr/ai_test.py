import os
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt


def validate_file(file_path, file_description):
    """파일 존재 여부를 검증하는 함수"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_description} 파일을 찾을 수 없습니다: {file_path}")


def predict_and_visualize(model_path, test_image_path, output_path="result_prediction.jpg", show_result=False):
    """
    학습된 YOLO 모델로 이미지를 테스트하고 예측 결과를 시각화합니다.

    Args:
        model_path (str): 학습된 YOLO 모델 파일 경로.
        test_image_path (str): 테스트 이미지 파일 경로.
        output_path (str): 예측 결과 이미지 저장 경로 (기본값: "result_prediction.jpg").
        show_result (bool): 예측 결과를 화면에 표시할지 여부 (기본값: False).

    Returns:
        None
    """
    # 1. 파일 검증
    validate_file(model_path, "모델")
    validate_file(test_image_path, "테스트 이미지")

    print("모델과 이미지를 로드 중입니다...")
    model = YOLO(model_path)

    # OpenCV로 이미지 읽기 (RGB로 변환)
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {test_image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 예측 수행
    print("모델 추론을 수행 중입니다...")
    results = model.predict(source=test_image_path, conf=0.35)

    # 3. 예측 결과 처리
    print("예측 결과를 처리 중입니다...")
    for i, result in enumerate(results):
        # 예측된 이미지를 시각화
        result_image = result.plot()

        # 결과 저장
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"결과 이미지가 저장되었습니다: {output_path}")

        # 출력 객체 정보
        boxes = result.boxes
        print(f"감지된 객체 수: {len(boxes)}")
        for j, box in enumerate(boxes):
            cls = int(box.cls[0])  # 클래스 ID
            conf = float(box.conf[0])  # 신뢰도
            label = model.names[cls]  # 클래스 이름
            print(f"  객체 {j + 1}: {label} (신뢰도: {conf:.2f})")

    # 4. 시각화 옵션
    if show_result:
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.axis("off")
        plt.title("Prediction Result")
        plt.show()


if __name__ == "__main__":
    # 사용자 입력 경로
    model_path = r"models/best.pt"  # 학습된 모델 경로
    test_image_path = r"image_모음/test_image/58.jpg"  # 테스트할 이미지 경로
    output_image_path = r"image_모음/결과_image/result.png"  # 예측 결과 이미지 저장 경로

    try:
        # 모델 테스트 실행
        predict_and_visualize(model_path, test_image_path, output_path=output_image_path, show_result=True)
    except Exception as e:
        print(f"오류 발생: {e}")
