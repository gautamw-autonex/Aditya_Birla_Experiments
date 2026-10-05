# Scripts

Runnable, flat-layout pipeline scripts. Run from repo root, e.g.
`python scripts/final_video.py`.

| Script | Purpose |
|---|---|
| `analyze_trail_videos.py` | Pairwise similarity analysis of trail videos |
| `final_video.py` | Build final video from similarity results |
| `rotate_videos.py` | Rotate H.264 videos 90° clockwise (ffmpeg) |
| `demo_logo_detection.py` | Logo-detection pipeline demo (needs `paint_box` package) |
| `color_fix.py`, `fix_rgb.py` | Color-correction helpers |
| `fix_frames_v2.py`, `fix_saved_frames.py` | Frame-repair helpers |
| `hardware_test_camera.py` | Raspberry Pi camera check — needs real `picamera2` hardware, not run in CI |

> Note: several scripts contain absolute paths to the original dev machine.
> Pass paths via argv/env or edit the `CONFIG` block at the top of each file.
