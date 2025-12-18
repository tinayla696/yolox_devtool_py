# 実行方法

## 準備：重みファイルの配置

YOLOXの学習済みモデル（`.pth`）をダウンロードし、`models/` ディレクトリに配置してください。

- **モデルの種類**: ファイル名に `yolox_s`, `yolox_m`, `yolox_l`, `yolox_x` を含めることで、ツールがモデルサイズを自動判別します。

```bash
# 配置例
models/
  ├── yolox_s.pth
  └── yolox_m.pth
```

## 1. 動画トラッキング（Track Tool）

動画内の車両と人物を検出し、IDを付与して追跡します。

### 基本コマンド：

```bash
# プロジェクトルートで実行（-m オプションを使用）
python3 -m src.track_tool -input <入力パス> --weights <重みファイルパス>
```

### 実行例

!!! note "単一ファイルの処理"
    特定の動画ファイルのみを処理します。
    ```bash
    python3 -m src.track_tool -i inputs/test_video.mp4 -w models/yolox_m.pth
    ```

!!! note "ディレクトリ一括処理"
    `inputs/`内にあるすべての動画ファイルを順次処理します。
    ```bash
    python3 -m src.track_tool -i inputs/ -w models/yolox_m.pth
    ```

!!! note "CPUで実行"
    GPU環境デモ強制的にCPUを使用する場合。  
    ```bash
    python3 -m src.track_tool -i inputs/ -w models/yolox_m.pth --device cpu
    ```

## 2. 画像解析（Image Tool）

ディレクトリ内の静止画を一括で物体検出し、画像とCSVレポートを出力します。

```bash
python3 -m src.image_tool -i inputs/ -o outputs/ -w models/yolox_m.pth
```

### オプション引数一覧

| 引数 | 説明 | デフォルト値 |
|------|------|--------------|
| `-i`, `--input` | 入力画像・動画のディレクトリまたはファイルパス | `inputs` |
| `-o`, `--output` | 出力ディレクトリのパス | `outputs` |
| `-w`, `--weights` | 学習済みモデルの重みファイルパス | `models/yolox_m.pth` |
| `-c`, `--confidence` | 検出信頼度の閾値 | `0.25` (動画は `0.1`) |
| `--device` | 使用デバイス（'auto', 'cpu', 'cuda'） | `auto` |