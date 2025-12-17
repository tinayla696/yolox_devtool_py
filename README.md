# 🚗 駐車場 動向計測 PoC ツール

## 概要

本プロジェクトは、YOLOX + ByteTrack を利用した駐車場内の車両・人物の動向計測システムの実現可能性を検証（PoC）するためのツールです。

主な目的は、カメラの設置画角や環境（昼夜・天候）による検出・追跡精度を、録画映像を使って事前に評価することです。

## 主な機能

* 入力動画ファイルに対する物体検出（YOLOX）と追跡（ByteTrack）の実行
* 追跡結果（ID、バウンディングボックス）を描画した動画ファイルの出力
* CPU / GPU の実行環境をコマンドライン引数で切り替え可能

## ディレクトリ構成

```tree
yolox_proctool_py/
├── .gitignore
├── README.md            # (このファイル)
├── mkdocs.yml           # MkDocs 設定
├── requirements.txt     # PoCツールの依存ライブラリ
├── docs/                # ドキュメント
├── input_videos/        # 入力動画
├── output_videos/       # 出力動画
├── models/              # YOLOX 重みファイル (.pth)
│   ├── yolox_m.pth
│   └── .gitkeep
└── src/                 # ソースコード
    └── track_tool.py    # メインスクリプト
```

## クイックスタート (実行例)

1. 仮想環境を有効化し、必要なライブラリをインストールします。（詳細は`docs/setup.md`を参照）
2. モデルを`models/`に、動画を`input_videos/`に配置します。
3. 以下のコマンドで推論を実行します。

```bash
python -m venv .venv
source .venv/bin/activate
# Windowsの場合は、 .venv/Scripts/activate
sudo apt update
sudo apt install -y build-essential python3-dev
pip install -r requirements.txt

# CPU版 PyTorch のインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# OR
# GPU版 PyTorch のインストール例 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# YOLOXのクローンとインストール
cd ./models
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
git clone https://github.com/ifzhang/ByteTrack.git
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
cd YOLOX
pip install -r requirements.txt
pip insatll cython-bbox
python setup.py develop
cd ../../
cp -r ./models/ByteTrack/yolox/tracker ./src/

# CPUを強制的に指定する場合
python -m src.track_tool --input input_videos/test.mp4 --output output_videos/ --weights models/yolox_m.pth --device cpu
# OR
# GPUを指定して実行
python -m src.track_tool --input input_videos/test.mp4 --output output_videos/ --weights models/yolox_m.pth --device cuda
```

## 5. ドキュメント

詳細な環境構築手順やツールの使用方法は、MkDocsのドキュメントを参照してください  
以下のコマンドでローカルサーバーを起動できます

```bash
mkdocs serve
```

ブラウザで `http://localhost:8000` にアクセスしてください
