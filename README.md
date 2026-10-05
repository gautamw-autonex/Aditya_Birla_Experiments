# Aditya Birla Paint Box Experiments

Computer-vision experiments for inspecting Aditya Birla paint boxes:
**logo detection** on curved cans, **QR / manufacturing-detail video checks**,
and color/frame fix-ups for trail videos.

Branch: `aditya-birla` ·
Remote: `gautamw-autonex/Aditya_Birla_Experiments`

## What lives here

- **Logo detection** — Faster R-CNN (MobileNetV3-Large-FPN) and RT-DETR
  fine-tunes for the Aditya Birla emblem; YOLOv8 dataset prep; curvature
  augmentation experiments for curved-can surfaces.
- **QR / ROI scanning** — OpenCV `QRCodeDetector` scanner (`notebooks/`) plus
  PaddleX experiments.
- **Trail-video pipeline** — rotate raw H.264 captures → pairwise similarity
  analysis → assemble final video → color/frame fix-ups (`scripts/`).
- **Trained weights** — `rtdetr_paint_box_model/` (RT-DETR, LFS).

Reference run: `notebooks/experiment.json` — Faster R-CNN
`mobilenet_v3_large_fpn`, 15 epochs, seed 42, CUDA (started 2026-09-28).

## Layout

```
scripts/                  Runnable pipeline (run from repo root)
  analyze_trail_videos.py   Sample 12 frames/video (256x342), pairwise
                            similarity → similarity_analysis/all_video_pairs.csv
  final_video.py            Assemble final video from the similarity CSV
  rotate_videos.py          Rotate mfg/qrcode H.264 90° CW (ffmpeg, near-lossless)
  color_fix.py / fix_rgb.py Color-correction passes over final_video frames
  fix_frames_v2.py / fix_saved_frames.py
                            Frame-repair passes (ProcessPool)
  demo_logo_detection.py    Curvature-augmentation demo (see caveat below)
  hardware_test_camera.py   Raspberry Pi check (picamera2, 1456x1088 RGB888)
tests/                    Pytest smoke tests (no hardware needed)
notebooks/
  logo_detection.ipynb      Rotation check → Faster R-CNN training → frame
                            extraction/sampling → Roboflow upload prep → YOLOv8 data
  logo_detection_v2.ipynb   RT-DETR fine-tune (RTDetrImageProcessor)
  kaggle_notebook/          OCR, Fast R-CNN MobileNetV3, RT-DETR forks (v3/v5),
                            global-shutter camera check, QR code
  qr_code_scanner.py        Standalone OpenCV QR + ROI scanner
  paddlex.py                PaddleX experiments
  rasberry/                 Pi trail captures (trail__videos/), analysis script
  aditya_birla_paint_box_logo_dataset/
                            Extracted training frames
  experiment.json           Reference training run config + epoch metrics
logo/video/               Logo-check videos (Git LFS: *.h264, *.mp4)
mfg_details/              Mfg-detail videos + final_videos/ (Git LFS)
qrcode/                   QR / AR-code videos + final_videos/ (Git LFS)
rtdetr_paint_box_model/   config.json, preprocessor_config.json,
                          model.safetensors ~164 MB (Git LFS)
assets/samples/           Sample images (git-ignored: *.jpg / *.png)
data/                     Datasets (git-ignored)
```

Large binaries (`*.h264`, `*.mp4`, `*.safetensors`, `*.tar.gz`, … — full list in
`.gitattributes`) are stored with [Git LFS](https://git-lfs.com):

```bash
git lfs install
git lfs pull
```

## Quickstart

```bash
poetry install
poetry run pytest tests/ -v
python scripts/rotate_videos.py      # straighten raw captures
python scripts/analyze_trail_videos.py
python scripts/final_video.py
```

## Data flow

```
Pi capture → rotate_videos → analyze_trail_videos (similarity CSV)
→ final_video → color_fix / fix_rgb / fix_frames_v2
```

```
Training: extract/sampling (notebook) → Roboflow upload images
→ Faster R-CNN / RT-DETR fine-tune → experiment.json metrics
→ rtdetr_paint_box_model/
```

## Caveats

- Several scripts hard-code absolute paths to the original dev machine
  (`/home/gautamw7/...`). Edit the `CONFIG`/`SRC` block at the top of each
  file or adapt to argv/env before running elsewhere.
- `demo_logo_detection.py` imports a `paint_box.logo_detection` package that
  does not exist in this repo (`pyproject.toml`/`Makefile` also reference a
  `src/paint_box` layout) — packaging that module is pending work.
- `hardware_test_camera.py` needs a real Raspberry Pi camera; it is kept out
  of `tests/` so CI stays green.
- `notebooks/` (`.ipynb`), `data/`, and sample images are git-ignored and
  live only on local disks — they are **not** on this branch's remote.
