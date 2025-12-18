# src/image_tool.py

import argparse
import os
import cv2
import torch
import numpy as np
import pandas as pd

# YOLOXライブラリからのインポート
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

# 判別対象クラス（COCOデータセット準拠：　0: person, 2: vehicle)
TARGET_CLASSES = [ 0, 2 ]
CLASS_NAMES = { 0: "person", 2: "vehicle" }

# ファイルパスからモデル名（yolox_s, yolox_m 等）を推定する関数
def get_exp_name_from_path(weights_path: str) -> str:
    """
    重みファイルのパスからモデル名（yolox_s, m, l, x）を判定する
    判定できない場合はデフォルトとして 'yolox_m' を返す
    """
    filename = os.path.basename(weights_path).lower()
    
    # チェックするモデルタイプのリスト（優先度順）
    known_models = ["yolox_s", "yolox_m", "yolox_l", "yolox_x", "yolox_tiny", "yolox_nano"]
    
    for model_name in known_models:
        if model_name in filename:
            print(f"Auto-detected model type: {model_name}")
            return model_name
            
    print("Warning: Could not infer model type from filename. Defaulting to 'yolox_m'.")
    return "yolox_m"

# 引数パーサーの設定
def get_argumentparser() -> argparse.ArgumentParser:
    """
    引数パーサーを設定する関数
    返り値: argparse.ArgumentParser オブジェクト
    """
    parser = argparse.ArgumentParser("YOLOX Image Processing & Reporting Tool")
    parser.add_argument("--input", "-i", type=str, default="inputs", help="Path to input image directory")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--weights", "-w", type=str, default="models/yolox_m.pth", help="Path to model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda'")
    return parser


# IDに基づいた色生成
def get_color(idx: int) -> tuple:
    """
    IDに基づいて一意な色を生成する関数
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

# 画像にバウンディングボックスと情報を描画する
def plot_boxes(image, detections, class_names) -> np.ndarray:
    """
    画像にバウンディングボックスとクラス情報を描画する関数
    """
    im = np.copy(image)
    text_scale = max(1, image.shape[1] //1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] // 500.))

    for i, det in enumerate(detections):
        x1,y1, x2, y2 = map(int, det[:4])
        score = det[4] * det[5]
        cls_id = int(det[6])
        obj_id = i + 1

        cls_name = class_names.get(cls_id, "Unknown")
        text = f"ID*{obj_id} {cls_name} {score: 2f}"

        color = get_color(obj_id)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, line_thickness)

        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)
        cv2.rectangle(im, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(im, text, (x1, y1 -4), cv2.FONT_HERSHEY_PLAIN, text_scale, (255,255,255), text_thickness)

    return im

# メイン関数
def main(args):
    """
    メイン処理関数
    """
    # 1. ディレクトリ準備
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "analysis_report.csv")

    # 2. デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # 3. モデルロード
    exp_name = get_exp_name_from_path(args.weights)
    exp = get_exp(None, exp_name)
    model = exp.get_model()
    model.to(device)
    model.eval()

    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 4. 画像リスト取得
    valid_exts = ( '.jpg', '.jpeg', '.png', ".bmp")
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found.")
        return
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images in {args.input}")

    # 解析結果を格納するリスト
    all_results = []
    pre_transform = ValTransform(legacy=False)

    # 5. 画像処理ループ
    for img_file in image_files:
        img_path = os.path.join(args.input, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"DEBUG: Could not read image {img_path}, skipping.")
            continue

        height, width = frame.shape[:2]

        # 前処理
        img, _ = pre_transform(frame, None, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).to(device)

        # 推論
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, exp.num_classes, args.confidence, 0.45, class_agnostic=True)

        detections_to_save = []
        if outputs[0] is not None:
            output_results = outputs[0].cpu().numpy()

            # 対象クラスのみ抽出
            mask = np.isin(output_results[:, 6], TARGET_CLASSES)
            detections = output_results[mask]

            # 座標補正
            ratio = min(exp.test_size[0] / height, exp.test_size[1] / width)
            detections[:, :4] /= ratio

            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det[:4]
                score = det[4] * det[5]
                cls_id = int(det[6])
                obj_id = i + 1

                # Pandas用データ作成（辞書型）
                result_data = {
                    "filename": img_file,
                    "object_id": obj_id,
                    "class": CLASS_NAMES.get(cls_id, "Unknown"),
                    "confidence_score": float(f"{score:.4f}"),
                    "bbox_x1": int(x1),
                    "bbox_y1": int(y1),
                    "bbox_x2": int(x2),
                    "bbox_y2": int(y2)
                }
                all_results.append(result_data)
                detections_to_save.append(det)

            # 描画と保存
            result_img = plot_boxes(frame, detections_to_save, CLASS_NAMES)
            save_path = os.path.join(args.output, img_file)
            cv2.imwrite(save_path, result_img)
            print(f"Processed: {img_file} -> {len(detections_to_save)} objects.")

    # 6. Pandasでレポート作成と保存
    if all_results:
        df = pd.DataFrame(all_results)
        # カラムの順序の整理（任意）
        df = df[["filename", "object_id", "class", "confidence_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        df.to_csv(report_path, index=False, encoding='utf-8')
        print(f"\nDone! Report saved to {report_path}")
        print(f"Total objects detected: {len(df)}")
    else:
        print("\nNo objects detected across all images. Empty report generated.")
        pd.DataFrame(columns=["filename", "object_id", "class", "confidence_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]).to_csv(report_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    args = get_argumentparser().parse_args()
    main(args)