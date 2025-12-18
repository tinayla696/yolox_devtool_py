# インストールガイド

本ツールの環境構築手順を説明します。YOLOXのインストールには特殊な手順（PyTorchの先行インストール等）が必要です。以下の手順通りに進めてください。

## 前提条件

- **OS**: Linux (Ubuntu 20.04/22.04) または WSL2
- **Python**: 3.8+
- **CUDA**: GPUを使用する場合は必須 (推奨バージョン 11.8)

## インストール手順

### 1. 仮想環境の構築

プロジェクトのルートディレクトリで仮想環境を作成し、有効化します。

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windowsの場合: .venv\Scripts\activate を実行
```

### 2. PyTorchのインストール

YOLOXのインストールスクリプトは、コンパイル時にPyTorchが存在することを前提としています。  
`requirements.txt`での一括インストールは失敗することがあるため、**必ず先に手動でインストール**しててください。  

!!! note "GUC(CUDA 11.8)"
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

!!! note "CUPのみ"
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

### 3. YOLOXクローンと依存関係の修正

```bash
# YOLOXのリポジトリをクローン
cd ./modules
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX/
```

`onnx-simplifier`のパッケージ名の不一致エラーを防ぐため、`requirements.txt`を修正します。

1. `requirements.txt`をテキストエディタで開く
2. 以下の行を書き換えて保存してください。
        ```diff
        -onnx-simplifier==0.3.6
        +onnxsim
        ```

### 4.  YOLOX本体のインストール

`pip` のビルド買うk理機能を無効化（`--no-build-isolation` オプション）して、YOLOXをインストールします。  
これにより、手順２でインストールした PyTorchが正しく認識されます。

```bash
# YOLOXのプロジェクトルートで実行
pip install --no-build-isolation -e .
```

### 5. ByteTrackクローンとソースコードの配置

1. ByteTrackのリポジトリをクローンします。

    ```bash
    cd ./modules
    git clone https://github.com/ifzhang/ByteTrack.git
    ```

2. ByteTrackの`tracker`ディレクトリを本プロジェクトの`src/`ディレクトリにコピーします。

    ```bash
    cp -r ./ByteTrack/byte_tracker ./../../src/tracker
    ```

### 6. 重みファイルのインストール

YOLOXの公式リリースから学習済みモデルをダウンロードし、`models/`ディレクトリに配置してください。

```bash
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -P models/
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth -P models/
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -P models/
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth -P models/
```

### 7. その他依存ライブラリ

トラッキング機能（ByteTrack）やドキュメント生成に必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
pip isntall scipy lapx mkdocs mkdocs-material
```

!!! info "Windows/WSL環境の方へ"
    `lapx`のインストールでビルドエラーが発生する場合でも、本ツールのトラッキング機能を`scipy`のみで動作するように調整されています。

---

## トラブルシューティング

!!! warning "AssertionError: torch is required for pre-compiling ops"
    原因: 手順2（PyTorch先行インストール）または手順4（`--no-build-isolation`）が守られていません。
    対処: 一度環境を作り直すか、PyTorchが入っている状態で `pip install -e . --no-build-isolation` を再実行してください。

!!! warning "Metadata for project name onnxsim..."
    原因: `requirements.txt` の記述が古いです。
    対処: 手順3に従い、`onnx-simplifier` を `onnxsim` に書き換えてください。