# ホーム

## プロジェクト概要

本プロジェクトは、YOLOXを使用した駐車場（車両・人物）向けの画像解析および動画トラッキングツールです。  
画像内の物体検出と、ByteTrackアルゴリズムを用いた動画内の追跡（ID付与）を行います。

### 主な機能

* **画像解析 (`src.image_tool`)**
    * 指定ディレクトリ内の静止画を一括解析。
    * バウンディングボックスの描画と、CSVレポートの出力。
* **動画トラッキング (`src.track_tool`)**
    * **ByteTrack** を統合し、高精度な追跡を実現。
    * 動画ファイル単体、またはディレクトリ内の一括処理に対応。
    * 検出結果（ID、クラス、スコア）を動画にオーバーレイ保存。

## ディレクトリ構成

```text
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
└── setup.py              # YOLOXインストール用
```

---
