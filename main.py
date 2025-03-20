import argparse
import os
from ultralytics import YOLO
from datetime import datetime


# Argument Parser 설정
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 및 하이퍼파라미터 튜닝 스크립트")

    default_data_yaml = os.path.join(os.getcwd(), "dataset", "data.yaml")
    parser.add_argument("--data_yaml", type=str, default=default_data_yaml, help="데이터셋 정보를 포함한 YAML 파일 경로")

    default_model_weights = os.path.join(os.getcwd(), "yolo11n.pt")
    parser.add_argument("--model_weights", type=str, default=default_model_weights, help="사전 학습된 모델 가중치 파일 경로")

    default_project_name = os.path.join(os.getcwd(), "runs", "tuning")
    parser.add_argument("--project_name", type=str, default=default_project_name, help="학습 결과를 저장할 최상위 경로")

    parser.add_argument("--num_trials", type=int, default=1, help="하이퍼파라미터 최적화를 위한 실험 반복 횟수")
    parser.add_argument("--epochs", type=int, default=50, help="학습 반복 횟수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--img_size", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "AdamW"], help="옵티마이저 선택")

    return parser.parse_args()


# YOLO 학습 함수
def train_yolo(args):
    print(f"데이터 YAML 파일 경로: {args.data_yaml}")
    print(f"모델 가중치 파일 경로: {args.model_weights}")
    print(f"프로젝트 저장 기본 경로: {args.project_name}")

    os.makedirs(args.project_name, exist_ok=True)  # 프로젝트 경로 생성

    model = YOLO(args.model_weights)

    if args.num_trials > 1:
        print("하이퍼파라미터 튜닝 시작...")
        results = model.tune(
            data=args.data_yaml,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            project=args.project_name,
            name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            optimizer=args.optimizer,
            patience=10,
            iterations=args.num_trials
        )
    else:
        print("YOLO 모델 학습 시작...")
        results = model.train(
            data=args.data_yaml,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            project=args.project_name,
            name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            optimizer=args.optimizer,
            patience=10
        )

    print("YOLO 모델 학습이 완료되었습니다!")

    # 학습된 모델 저장
    save_dir = r"C:\Users\dnjs8\IdeaProjects\aiaiaiaiaiaiaia\scr\models"  # 저장 경로
    os.makedirs(save_dir, exist_ok=True)  # 저장 폴더 생성

    # 저장 파일명 생성
    save_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.torchscript.pt"
    model.export(format="torchscript", name=os.path.join(save_dir, save_name))

    print(f"학습된 모델이 {os.path.join(save_dir, save_name)}에 저장되었습니다.")

    return results


# main() 함수
def main():
    args = parse_arguments()

    args.data_yaml = os.path.abspath(args.data_yaml)
    args.model_weights = os.path.abspath(args.model_weights)
    args.project_name = os.path.abspath(args.project_name)

    train_yolo(args)


if __name__ == "__main__":
    main()
