"""Smoke test — no hardware or model weights needed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_scripts_present():
    scripts = ROOT / "scripts"
    assert scripts.is_dir()
    expected = {
        "analyze_trail_videos.py",
        "final_video.py",
        "rotate_videos.py",
        "demo_logo_detection.py",
    }
    present = {p.name for p in scripts.glob("*.py")}
    assert expected <= present


def test_lfs_attributes_present():
    gitattributes = ROOT / ".gitattributes"
    assert gitattributes.exists()
    text = gitattributes.read_text()
    assert "filter=lfs" in text
