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

### 3. YOLOX依存関係の修正

`onnx-simplifier`のパッケージ名の不一致エラーを防ぐため、`requirements.txt`を修正します。

1. `requirements.txt`をテキストエディタで開く
2. 以下の行を書き換えて保存してください。
        ```diff
        -onnx-simplifier==0.3.6
        +onnx-simplifier==0.3.7
        ```

### 4.  YOLOXほんたいのインストール

`pip` のビルド買うk理機能を無効化（`--no-build-isolation` オプション）して、YOLOXをインストールします。  
これにより、手順２でインストールした PyTorchが正しく認識されます。

```bash
# プロジェクトルートで実行
pip install --no-build-isolation -e .
```

### 5. その他依存ライブラリ

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