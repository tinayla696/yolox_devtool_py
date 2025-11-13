# src/track_tool.py
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
# from yolox.utils.visualize import plot_tracking  <-- ★削除しました
from src.tracker.byte_tracker import BYTETracker

def get_argument_parser():
    parser = argparse.ArgumentParser("YOLOX + ByteTrack PoC Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default="output_videos", help="Output directory")
    parser.add_argument("--weights", "-w", type=str, default="models/yolox_m.pth", help="Path to model weights")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Tracking threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda'")
    return parser

# ★追加: 色を決める関数
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

# ★追加: トラッキング結果を描画する関数 (YOLOXのデモより移植)
def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

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
    class TrackerArgs:
        def __init__(self):
            self.track_thresh = args.track_thresh
            self.track_buffer = 30
            self.match_thresh = 0.8
            self.mot20 = False
    
    tracker = BYTETracker(TrackerArgs(), frame_rate=30)

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
            keep = (output_results[:, 6] == 0) | (output_results[:, 6] == 2)
            detections = output_results[keep]
            
            # ByteTrack用にフォーマット整形
            final_scores = detections[:, 4] * detections[:, 5]
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

        # 自前で定義した関数を使用
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