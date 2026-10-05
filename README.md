# Aditya Birla Paint Box Experiments

Computer-vision experiments for paint-box inspection: logo detection,
QR / manufacturing-detail video checks, and color/frame fix-up utilities.

Branch: `aditya-birla` ·
Remote: `gautamw-autonex/Aditya_Birla_Experiments`

## Layout

```
scripts/                  Runnable pipeline scripts (see table below)
  analyze_trail_videos.py   Pairwise similarity analysis of trail videos
  final_video.py            Build final video from similarity results
  rotate_videos.py          Rotate H.264 videos 90° clockwise (ffmpeg)
  demo_logo_detection.py    Logo-detection pipeline demo
  color_fix.py / fix_rgb.py Frame color-correction helpers
  fix_frames_v2.py / fix_saved_frames.py
                            Frame-repair helpers
  hardware_test_camera.py   Raspberry Pi camera check (needs picamera2 hardware)
tests/                    Pytest smoke tests (no hardware needed)
assets/samples/           Sample images (git-ignored: *.jpg / *.png)
logo/video/               Logo-check videos (Git LFS: *.h264, *.mp4)
mfg_details/              Manufacturing-detail videos, incl. final_videos/ (Git LFS)
qrcode/                   QR / AR-code videos, incl. final_videos/ (Git LFS)
rtdetr_paint_box_model/   Detection model weights (Git LFS: *.safetensors)
notebooks/                Exploration notebooks (git-ignored: *.ipynb)
data/                     Datasets (git-ignored)
```

Large binaries (`*.h264`, `*.mp4`, `*.safetensors`, `*.tar.gz`, … — see
`.gitattributes`) are stored with [Git LFS](https://git-lfs.com):

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

## Notes

- Several scripts still contain absolute paths to the original dev machine.
  Pass paths via argv/env or edit the `CONFIG` block at the top of each file.
- `hardware_test_camera.py` requires a real Raspberry Pi camera (`picamera2`);
  it is intentionally kept out of `tests/` so CI stays green.
- `pyproject.toml` / `Makefile` still reference a `src/paint_box` package that
  does not exist yet — packaging that layout is pending work.
