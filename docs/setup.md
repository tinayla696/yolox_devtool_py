# 環境構築 (setup.md)

## 仮想環境の構築

WSL2 (Ubuntu) 上で、Python 3.8〜3.10 の仮想環境を構築・有効化します。

```bash
python3 -m venv venv
source venv/bin/activate
```

## YOLOXのセットアップ

`track_tool.py` は、公式のYOLOXライブラリに依存しています  
以下の手順でYOLOXをインストールしてください

```bash
# 1. YOLOXの公式リポジトリをクローン
git clone [https://github.com/Megvii-BaseDetection/YOLOX.git](https://github.com/Megvii-BaseDetection/YOLOX.git)

# 2. クローンしたディレクトリに移動
cd YOLOX

# 3. YOLOXの基本的な依存ライブラリをインストール
pip install -r requirements.txt
```

## PyTorchのインストール（最重要）

YOLOX（および`track_tools.py）を実行するPCの環境に合わせて、PyTorchをインストールします。  

### 【A】NVIDIA GPU があるPCの場合（WSL2）
    1. WSL2でCUDAが利用可能であることを確認してください。
    2. ご自身のCUDAバージョンにあったPyTorchをインストールします。（例：CUDA 11.8の場合）

    ```bash
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

### 【B】NVIDIA GPU がないPCの場合（CPUのみ）

    CPU版のPyTorchをインストールします。

    ```bash
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    ```

## 依存ライブラリをYOLOX本体のインストール

YOLOXとByteTrackの実行に必要な残りのライブラリをインストールします。

```bash
# 1. ByteTrackの高速化に必要なライブラリ
pip install cython-bbox

# 2. YOLOXを開発モードでインストール
python setup.py develop
```

## PoCツールのライブラリインストール

`yolox_proctool_py` のルートディレクトリに戻り、MkDocsなどこのPoCツール自体に必要なライブラリをインストールします。

```bash
cd ../
pip install -r requirements.txt
```

!!! Note "参考情報"
    `yolox_proctool_py/requirements.txt` には以下を記載しておきます。
    ```text
    mkdocs
    mkdocs-material
    opencv-python
    ```
    (`torch` や `yolox` 関連は上記手順でインストール済みのため、記載不要です)