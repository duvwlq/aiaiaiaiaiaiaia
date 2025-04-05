import argparse
import os
from ultralytics import YOLO
from datetime import datetime
import wandb
import cv2
import yaml
from sklearn.model_selection import KFold
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Argument Parser 설정
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO 모델 학습 및 성능 최적화 스크립트")

    default_data_yaml = os.path.join(os.getcwd(), "dataset", "data.yaml")
    parser.add_argument("--data_yaml", type=str, default=default_data_yaml, help="데이터셋 정보를 포함한 YAML 파일 경로")

    default_model_weights = os.path.join(os.getcwd(), "yolo11n.pt")
    parser.add_argument("--model_weights", type=str, default=default_model_weights, help="사전 학습된 모델 가중치 파일 경로")

    default_project_name = os.path.join(os.getcwd(), "runs", "tuning")
    parser.add_argument("--project_name", type=str, default=default_project_name, help="학습 결과를 저장할 경로")

    parser.add_argument("--num_trials", type=int, default=1, help="하이퍼파라미터 최적화를 위한 실험 횟수")
    parser.add_argument("--epochs", type=int, default=50, help="학습 반복 횟수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--img_size", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "AdamW"], help="옵티마이저 선택")
    parser.add_argument("--k_folds", type=int, default=1, help="K-Fold Cross-validation 개수")
    parser.add_argument("--use_amp", action="store_true", help="Mixed Precision Training 활성화 여부 (FP16)")

    return parser.parse_args()


# 데이터 증강 함수 (OpenCV 기반 대체)
def augment_image(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    if random.random() < 0.2:
        alpha = 1.0 + random.uniform(-0.2, 0.2)
        beta = random.randint(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        rows, cols, _ = image.shape
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)

        dx = random.randint(-int(0.0625 * cols), int(0.0625 * cols))
        dy = random.randint(-int(0.0625 * rows), int(0.0625 * rows))
        M[0, 2] += dx
        M[1, 2] += dy

        image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)

    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = cv2.resize(image, (640, 640))

    return image.astype(np.float32)


# K-Fold Cross-validation 수행
def perform_kfold_cv(data_yaml, k, args):
    with open(data_yaml, 'r', encoding='utf-8') as file:
        data_config = yaml.safe_load(file)

    dataset_path = data_config.get('path')
    if not dataset_path:
        print("데이터 경로가 없습니다. YAML 파일을 확인해주세요.")
        return

    images_path = os.path.join(dataset_path, 'images/train')
    images = os.listdir(images_path)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"\n=== Fold {fold + 1}/{k} 시작 ===")
        fold_path = os.path.join(dataset_path, f"fold_{fold}")
        os.makedirs(fold_path, exist_ok=True)

        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]

        new_yaml_path = os.path.join(fold_path, f"fold_{fold}_data.yaml")
        with open(new_yaml_path, 'w', encoding='utf-8') as yaml_file:
            new_yaml = data_config.copy()
            new_yaml['train'] = train_images
            new_yaml['val'] = val_images
            yaml.dump(new_yaml, yaml_file, allow_unicode=True)

        args.data_yaml = new_yaml_path
        train_yolo(args)

    print(f"K-Fold Cross-validation ({k}-fold) 완료!")


# YOLO 학습 수행
def train_yolo(args):
    # wandb API Key 설정 (환경 변수로부터 가져오기)
    try:
        api_key = os.environ.get("WANDB_API_KEY", 'API')
        if not api_key:
            raise ValueError("WANDB_API_KEY가 설정되지 않았습니다. export WANDB_API_KEY로 설정해주세요.")
        wandb.login(key=api_key)
    except Exception as e:
        print(f"[ERROR] wandb 로그인 실패: {e}")
        return

    wandb.init(project="YOLO_Optimization_Project", name=f"YOLO_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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
            optimizer=args.optimizer
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
            optimizer=args.optimizer
        )

    # JSON 직렬화 가능한 Dict로 변환하여 wandb 기록
    results_dict = {}
    if hasattr(results, 'metrics'):
        metrics = results.metrics
        results_dict = {
            'precision': metrics.get('precision', None),
            'recall': metrics.get('recall', None),
            'mAP': metrics.get('map', None),
            'mAP_50': metrics.get('map_50', None),
            'mAP_75': metrics.get('map_75', None),
            'box_loss': metrics.get('box_loss', None),
            'cls_loss': metrics.get('cls_loss', None),
            'obj_loss': metrics.get('obj_loss', None)
        }
    else:
        print("[Warning] `results` 객체에서 메트릭스를 찾을 수 없습니다. 확인이 필요합니다.")

    wandb.log({"results": results_dict})
    wandb.finish()

    print("YOLO 학습 완료!")

    save_dir = r"C:\Users\dnjs8\IdeaProjects\aiaiaiaiaiaiaia\scr\models"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.torchscript.pt"
    model.export(format="torchscript", name=os.path.join(save_dir, save_name))

    print(f"학습된 모델이 {os.path.join(save_dir, save_name)}에 저장되었습니다.")
    return results



# 학습 결과 시각화
def visualize_training_results(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Training log file not found at {log_file}")
        return

    try:
        log_data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error while reading the file: {e}")
        return

    sns.lineplot(data=log_data, x='epoch', y='loss', label='Training Loss')
    sns.lineplot(data=log_data, x='epoch', y='val_loss', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    sns.lineplot(data=log_data, x='epoch', y='mAP', label='mAP')
    plt.title("Mean Average Precision (mAP)")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.show()


# main() 함수
def main():
    args = parse_arguments()

    args.data_yaml = os.path.abspath(args.data_yaml)
    args.model_weights = os.path.abspath(args.model_weights)
    args.project_name = os.path.abspath(args.project_name)

    if args.k_folds > 1:
        print(f"K-Fold Cross-validation ({args.k_folds}-fold) 시작...")
        perform_kfold_cv(args.data_yaml, args.k_folds, args)
    else:
        train_yolo(args)

    log_file_path = os.path.join(args.project_name, "train_log.csv")
    visualize_training_results(log_file_path)


if __name__ == "__main__":
    main()
