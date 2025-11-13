# 実行方法

1. 仮想環境を有効化します。

    ```bash
    source venv/bin/activate
    ```

2. （推奨）YOLOXのモデルを `models/` に、動画を `input_videos/` に配置します。

3. `track_tool.py` を実行します。

**【注意】** `src` ディレクトリ構成のため、`-m` オプションを付けて実行します。

```bash
# GPUがある場合 (自動判別)
python -m src.track_tool --input input_videos/test_daytime.mp4 --output output_videos/ --weights models/yolox_m.pth

# CPUを強制的に指定する場合
python -m src.track_tool --input input_videos/test_night.mp4 --output output_videos/ --weights models/yolox_m.pth --device cpu
```
