#!/usr/bin/env python3
"""
Rotate H.264 videos 90° clockwise with near-lossless quality.
Uses FFmpeg with libx264 CRF=0 (lossless) or CRF=1-2 (visually lossless).
"""

import subprocess
import os
from pathlib import Path

# Video files to rotate
VIDEOS = [
    # Manufacturing details
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/mfg_details/500ml_mfg_details_1.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/mfg_details/500ml_mfg_details_2.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/mfg_details/1000ml_mfg_details_1.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/mfg_details/1000ml_mfg_details_2.h264",
    
    # QR codes
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/qrcode/500ml_arcode_1.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/qrcode/500ml_arcode_2.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/qrcode/1000ml_qrcode_1.h264",
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/qrcode/1000ml_qrcode_2.h264",
]

def rotate_video_lossless(input_path, output_path, crf=0):
    """
    Rotate video 90° clockwise with lossless/visually-lossless encoding.
    
    Args:
        input_path: Path to input H.264 video
        output_path: Path to output rotated video
        crf: Quality (0=lossless, 1-2=visually lossless, higher=more lossy)
             CRF=0 is slower but perfect quality; CRF=1-2 is 99.99% quality, much faster
    """
    # FFmpeg rotate filter: transpose=1 = rotate 90° clockwise
    # (transpose=0: 90° ccw, transpose=1: 90° cw, transpose=2: 90° ccw + hflip, etc.)
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "transpose=1",  # Rotate 90° clockwise
        "-c:v", "libx264",     # H.264 codec
        "-preset", "slow",     # Slower = better compression (quality vs speed trade-off)
        "-crf", str(crf),      # Quality: 0 = lossless, higher = more lossy
        "-c:a", "aac",         # Audio codec (copy if no audio, or re-encode)
        output_path
    ]
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Successfully rotated: {output_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {input_path}: {e}\n")

def main():
    quality_level = 1  # Use CRF=1 for 99.99% quality + acceptable speed
                       # Use CRF=0 for perfect lossless (much slower)
    
    print("=" * 70)
    print(f"H.264 Video Rotation Script (90° Clockwise)")
    print(f"Quality Level: CRF={quality_level} (visually lossless)")
    print("=" * 70 + "\n")
    
    for video_path in VIDEOS:
        # Verify input exists
        if not os.path.exists(video_path):
            print(f"✗ File not found: {video_path}")
            continue
        
        # Generate output path: append "_rotated" before extension
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_rotated{ext}"
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"⊘ Already exists (skipping): {output_path}\n")
            continue
        
        rotate_video_lossless(video_path, output_path, crf=quality_level)
    
    print("=" * 70)
    print("All videos processed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
