# 🚗 YOLOX Processing Tool

YOLOXを使用した車両・人物向けの画像解析および動画トラッキングツールです。  
画像内の物体検出と、ByteTrackアルゴリズムを用いた動画内の追跡（ID付与）を行います。

## 主な機能

- **画像解析 (`src.image_tool`)**:
    - 指定ディレクトリ内の画像を解析。
    - バウンディングボックスの描画と、CSVレポートの出力。
    - モデルサイズ（S, M, L, Xなど）を重みファイル名から自動判定。
- **動画トラッキング (`src.track_tool`)**:
    - **ByteTrack** を統合し、高精度な追跡を実現。
    - 動画ファイル単体、またはディレクトリ内の一括処理に対応。
    - 検出結果（ID、クラス、スコア）を動画にオーバーレイ保存。

## ディレクトリ構成

```tree
.
├── inputs/               # 解析対象の画像・動画を配置
├── outputs/              # 解析結果の保存先
├── models/               # 学習済み重みファイル (.pth) を配置
├── modules/              # 共通モジュール
│   ├── ByteTrack/        # ByteTrack関連モジュール ※ git clone で取得
│   └── YOLOX/            # YOLOX関連モジュール ※ git clone で取得
├── src/
│   ├── image_tool.py     # 画像解析用スクリプト
│   ├── track_tool.py     # 動画トラッキング用スクリプト
│   └── tracker/          # ByteTrack実装モジュール
├── requirements.txt      # 依存ライブラリ
└── README.md             # 本ファイル
```

## インストール

詳細な手順は `docs/installation.md` を参照してください。


### クイックスタート:

```bash
# 1. 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate
# Windowsの場合: .venv\Scripts\activate を実行

# 2. PyTorchのインストール（環境に合わせて変更してください）
pip install torch torchvision

# 3. YOLOXのインストール（ビルド隔離を無効化します）
cd ./modules
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX/
pip install --no-build-isolation -e .


git clone https://github.com/ifzhang/ByteTrack.git

# 4. その他依存ライブラリのインストール
pip install -r requirements.txt
```

## 使い方

### 1. 準備（重みファイルのダウンロード）

YOLOXの公式リリースから学習済みモデルをダウンロードし、`modules/`ディレクトリに配置してください

```bash
# 例: YOLOX-M と YOLOX-S をダウンロード
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth -P models/
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P models/
```

### 2. 画像解析（Image Tool）

`inputs/` ディレクトリ内の画像を一括解析し、`outputs/`ディレクトリに結果画像とCSVレポートを出力します。

```bash
# 基本的な実行（YOLO-M 使用）
python3 -m src.image_tool -i inputs -o outputs -w models/yolox_m.pth

# YOLO-S を使用する場合（重みファイルを変更する）
python3 -m src.image_tool -i inputs -o outputs -w models/yolox_s.pth
```

### 3. 動画トラッキング（Track Tool）

動画内の Person(0) と Car(2) を追跡し、IDを付与した動画を出力します。

#### 単一ファイルの処理：

```bash
python3 -m src.track_tool -i inputs/sample_movie.mp4 -w models/yolox_m.pth
```

#### ディレクトリ一括処理：

`inputs/`ディレクトリ内のすべての動画ファイルを順次処理します。

```bash
python3 -m src.track_tool -i inputs/ -w models/yolox_m.pth
```

## オブション引数

| 引数 | 説明 | デフォルト値 |
|------|------|--------------|
| `-i`, `--input` | 入力画像・動画のディレクトリまたはファイルパス | `inputs` |
| `-o`, `--output` | 出力ディレクトリのパス | `outputs` |
| `-w`, `--weights` | 学習済みモデルの重みファイルパス | `models/yolox_m.pth` |
| `-c`, `--confidence` | 検出信頼度の閾値 | `0.25` (動画は `0.1`) |
| `--device` | 使用デバイス（'auto', 'cpu', 'cuda'） | `auto` |

## license / Credits

- **YOLOX**: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)  
- **ByteTrack**: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

---
