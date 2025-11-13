# src/track_tool.py
import argparse
import os
import time
import cv2
import torch

# YOLOXライブラリからのインポート
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import ByteTracker  # YOLOX内のByteTrackを使用

def get_argument_parser():
    parser = argparse.ArgumentParser("YOLOX + ByteTrack PoC Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default="output_videos", help="Output directory")
    parser.add_argument("--weights", "-w", type=str, default="models/yolox_m.pth", help="Path to model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Tracking threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda'")
    return parser

def main(args):
    # 1. デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. モデルの準備
    exp = get_exp(None, "yolox-m")
    model = exp.get_model()
    model.to(device)
    model.eval()

    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 3. Trackerの初期化
    # (引数をオブジェクト形式で渡す必要があるため簡易クラスを作成)
    class TrackerArgs:
        def __init__(self):
            self.track_thresh = args.track_thresh
            self.track_buffer = 30
            self.match_thresh = 0.8
            self.mot20 = False
    
    tracker = ByteTracker(TrackerArgs(), frame_rate=30)

    # 4. 動画入力の準備
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, os.path.basename(args.input))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 5. 推論ループ
    pre_transform = ValTransform(legacy=False)
    frame_id = 0
    
    print("Start processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 前処理
        img, _ = pre_transform(frame, None, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)

        t0 = time.time()
        
        # 推論
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, exp.num_classes, args.confidence, 0.45, class_agnostic=True)

        # トラッキング
        online_targets = []
        if outputs[0] is not None:
            output_results = outputs[0].cpu().numpy()
            # Person(0) と Car(2) のみを抽出 (COCO)
            # detections format: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
            # ByteTracker expects: [x1, y1, x2, y2, score]
            
            keep = (output_results[:, 6] == 0) | (output_results[:, 6] == 2)
            detections = output_results[keep]
            
            # score = obj_conf * class_conf
            final_scores = detections[:, 4] * detections[:, 5]
            
            # ByteTracker用にフォーマット整形: x1, y1, x2, y2, score
            tracks_in = detections[:, :5]
            tracks_in[:, 4] = final_scores
            
            online_targets = tracker.update(tracks_in, [height, width], exp.test_size)

        # 描画
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
            online_scores.append(t.score)

        text_scale = 2
        text_thickness = 2
        line_thickness = 3
        
        online_im = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
        )
        
        vid_writer.write(online_im)
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"Processing frame {frame_id}...")

    cap.release()
    vid_writer.release()
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    args = get_argument_parser().parse_args()
    main(args)