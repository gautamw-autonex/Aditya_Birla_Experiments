# Aditya Birla Paint Box Experiments

Computer-vision experiments for paint-box inspection: logo detection,
QR/mfg-detail video checks, and color/fix-up utilities.

Branch: `aditya-birla`

## Layout

```
scripts/            Runnable pipeline scripts (video analysis, fixes, rotation, demos)
  hardware_test_camera.py   Raspberry Pi camera check (needs picamera2 hardware)
tests/              Pytest smoke tests (no hardware needed)
assets/samples/     Sample images (ignored by git: *.jpg/*.png)
logo/video/         Logo check videos (Git LFS: *.h264, *.mp4)
mfg_details/        Manufacturing-detail videos (Git LFS)
qrcode/             QR / AR-code videos (Git LFS)
rtdetr_paint_box_model/  Detection model weights (Git LFS: *.safetensors)
notebooks/          Exploration notebooks (ignored: *.ipynb)
data/               Datasets (ignored)
```

Large binaries (`*.h264`, `*.mp4`, `*.safetensors`, `*.tar.gz`) are stored
with [Git LFS](https://git-lfs.com). Install it and pull with:

```bash
git lfs install
git lfs pull
```

## Quickstart

```bash
poetry install
poetry run pytest tests/ -v
python scripts/rotate_videos.py
python scripts/final_video.py
```

See `scripts/README.md` for per-script notes.
