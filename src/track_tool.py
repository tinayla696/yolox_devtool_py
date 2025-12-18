import argparse
import os
import time
import cv2
import torch
import numpy as np

# YOLOXライブラリからのインポート
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

# 公式ByteTrackからのインポート
from src.tracker.byte_tracker import BYTETracker

# 判別対象クラス（COCO準拠: 0: person, 2: car）
TARGET_CLASSES = [0, 2]

def get_argument_parser():
    parser = argparse.ArgumentParser("YOLOX + ByteTrack Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video file or directory")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--weights", "-w", type=str, default="models/yolox_m.pth", help="Path to model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.1, help="Detection confidence threshold")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Tracking threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Buffer frames for lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda'")
    return parser

def get_exp_name_from_path(weights_path: str) -> str:
    filename = os.path.basename(weights_path).lower()
    known_models = ["yolox_s", "yolox_m", "yolox_l", "yolox_x", "yolox_tiny", "yolox_nano"]
    for model_name in known_models:
        if model_name in filename:
            print(f"Auto-detected model type: {model_name}")
            return model_name
    return "yolox_m"

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, online_targets, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(1, image.shape[1] // 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] // 500.))

    cv2.putText(im, f'Frame: {frame_id} FPS: {fps:.1f} Obj: {len(online_targets)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, text_scale * 1.5, (0, 255, 0), 2)

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        score = t.score
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h

        color = get_color(tid)
        label_text = f"ID:{tid} {score:.2f}"

        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)
        cv2.rectangle(im, (int(x1), int(y1) - text_h - 4), (int(x1) + text_w, int(y1)), color, -1)
        cv2.putText(im, label_text, (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 255, 255), text_thickness)

    return im

# --- 1つの動画を処理する関数 ---
def process_video(video_path, args, model, exp, device):
    video_name = os.path.basename(video_path)
    save_path = os.path.join(args.output, video_name.replace(".", "_tracked."))
    
    print(f"\n[START] Processing: {video_name}")

    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writer設定
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Trackerの初期化 (動画ごとに新しく作ることでIDをリセット)
    class TrackerArgs:
        def __init__(self):
            self.track_thresh = args.track_thresh
            self.track_buffer = args.track_buffer
            self.match_thresh = args.match_thresh
            self.mot20 = False
    
    tracker = BYTETracker(TrackerArgs(), frame_rate=30)
    
    pre_transform = ValTransform(legacy=False)
    frame_id = 0
    t0_overall = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 前処理
        img, _ = pre_transform(frame, None, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(device)

        t_start = time.time()
        
        # 推論
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, exp.num_classes, args.confidence, 0.45, class_agnostic=True)

        # トラッキング
        online_targets = []
        if outputs[0] is not None:
            output_results = outputs[0].cpu().numpy()
            keep = np.isin(output_results[:, 6], TARGET_CLASSES)
            detections = output_results[keep]
            
            if len(detections) > 0:
                final_scores = detections[:, 4] * detections[:, 5]
                tracks_in = detections[:, :5]
                tracks_in[:, 4] = final_scores
                online_targets = tracker.update(tracks_in, [height, width], exp.test_size)

        fps_curr = 1. / (time.time() - t_start)
        online_im = plot_tracking(frame, online_targets, frame_id=frame_id + 1, fps=fps_curr)
        vid_writer.write(online_im)
        
        frame_id += 1
        if frame_id % 50 == 0:
            print(f"Processing frame {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)", end="\r")

    cap.release()
    vid_writer.release()
    elapsed = time.time() - t0_overall
    print(f"\n[DONE] Saved to {save_path} (Time: {elapsed:.2f}s)")


def main(args):
    # 1. デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. モデルロード（1回だけ行う）
    exp_name = get_exp_name_from_path(args.weights)
    exp = get_exp(None, exp_name)
    model = exp.get_model()
    model.to(device)
    model.eval()
    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 3. 入力ファイルのリスト作成
    video_files = []
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')

    if os.path.isdir(args.input):
        # ディレクトリなら中の動画ファイルを探索
        files = sorted(os.listdir(args.input))
        for f in files:
            if f.lower().endswith(valid_exts):
                video_files.append(os.path.join(args.input, f))
        print(f"Found {len(video_files)} videos in directory '{args.input}'")
    
    elif os.path.isfile(args.input):
        # 単一ファイルならそれをリストに追加
        video_files.append(args.input)
    
    else:
        print(f"Error: input '{args.input}' is not found.")
        return

    os.makedirs(args.output, exist_ok=True)

    # 4. 順次処理
    for video_path in video_files:
        process_video(video_path, args, model, exp, device)

    print("\nAll videos processed.")

if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)