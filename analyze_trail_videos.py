import os
import re
import cv2
import math
import numpy as np
import pandas as pd

from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/notebooks/rasberry/trail__videos"
)

OUTPUT_DIR = VIDEO_DIR / "similarity_analysis"

FRAME_SAMPLES = 12

# Resize sampled frames before comparison
COMPARE_WIDTH = 256
COMPARE_HEIGHT = 342

# Candidate threshold.
# Higher = more likely to flag videos as similar.
CANDIDATE_PHASH_DISTANCE = 18

# Final similarity threshold.
HIGH_SIMILARITY_THRESHOLD = 0.90
VERY_HIGH_SIMILARITY_THRESHOLD = 0.95

# Number of workers
MAX_WORKERS = max(
    1,
    min(8, (os.cpu_count() or 4))
)


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FRAMES_DIR = OUTPUT_DIR / "sampled_frames"
SUSPICIOUS_DIR = OUTPUT_DIR / "suspicious_pairs"

FRAMES_DIR.mkdir(
    exist_ok=True
)

SUSPICIOUS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# VIDEO INDEX
# ============================================================

def get_video_index(path):

    match = re.search(
        r"video_(\d+)\.mp4$",
        path.name
    )

    if match:
        return int(match.group(1))

    return None


video_files = sorted(
    VIDEO_DIR.glob("video_*.mp4"),
    key=lambda p: (
        get_video_index(p)
        if get_video_index(p) is not None
        else 999999
    )
)


print()
print("=" * 100)
print("TRAIL VIDEO NEAR-DUPLICATE / SIMILARITY ANALYSIS")
print("=" * 100)
print()

print(f"Videos found : {len(video_files)}")
print(f"Samples/video: {FRAME_SAMPLES}")
print(f"Workers      : {MAX_WORKERS}")
print()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def phash(image, hash_size=16):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (
            hash_size * 4,
            hash_size * 4
        )
    )

    gray = np.float32(gray)

    dct = cv2.dct(gray)

    dct_low = dct[
        :hash_size,
        :hash_size
    ]

    median = np.median(
        dct_low
    )

    bits = dct_low > median

    return bits.flatten()


def hamming_distance(hash_a, hash_b):

    return int(
        np.count_nonzero(
            hash_a != hash_b
        )
    )


# ============================================================
# FRAME NORMALIZATION
# ============================================================

def normalize_frame(frame):

    if frame is None:
        return None

    # Convert to fixed size
    frame = cv2.resize(
        frame,
        (
            COMPARE_WIDTH,
            COMPARE_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    # Convert to LAB.
    # LAB is less sensitive to raw brightness differences
    # than direct RGB/BGR comparison.
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    return lab


# ============================================================
# FRAME SIMILARITY
# ============================================================

def pixel_similarity(frame_a, frame_b):

    if frame_a is None or frame_b is None:
        return 0.0

    a = normalize_frame(frame_a)
    b = normalize_frame(frame_b)

    diff = cv2.absdiff(
        a,
        b
    )

    mean_diff = np.mean(
        diff
    )

    # LAB channel maximum roughly 255.
    similarity = 1.0 - (
        mean_diff / 255.0
    )

    return float(
        np.clip(
            similarity,
            0.0,
            1.0
        )
    )


# ============================================================
# FRAME LOADING
# ============================================================

def sample_video(path):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        return {
            "path": str(path),
            "frames": [],
            "frame_indices": [],
            "frame_count": 0,
            "fps": 0,
            "duration": 0,
            "error": "Could not open video"
        }

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if frame_count <= 0:

        cap.release()

        return {
            "path": str(path),
            "frames": [],
            "frame_indices": [],
            "frame_count": 0,
            "fps": fps,
            "duration": 0,
            "error": "No frames"
        }

    # Sample evenly through entire video.
    indices = np.linspace(
        0,
        frame_count - 1,
        FRAME_SAMPLES,
        dtype=int
    )

    frames = []
    valid_indices = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ok, frame = cap.read()

        if not ok:
            continue

        frames.append(frame)
        valid_indices.append(
            int(idx)
        )

    cap.release()

    return {
        "path": str(path),
        "frames": frames,
        "frame_indices": valid_indices,
        "frame_count": frame_count,
        "fps": fps,
        "duration": (
            frame_count / fps
            if fps > 0
            else 0
        ),
        "error": None
    }


# ============================================================
# VIDEO FEATURE EXTRACTION
# ============================================================

def process_video(path):

    result = sample_video(
        path
    )

    frames = result["frames"]

    hashes = []

    for frame in frames:

        hashes.append(
            phash(frame)
        )

    result["hashes"] = hashes

    return result


# ============================================================
# LOAD ALL VIDEOS
# ============================================================

print("Sampling videos...")
print()

video_data = {}

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            process_video,
            path
        ): path
        for path in video_files
    }

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Videos"
    ):

        path = futures[future]

        try:

            result = future.result()

            video_data[path.name] = result

        except Exception as e:

            print(
                f"\nERROR: {path.name}: {e}"
            )


print()
print(
    f"Successfully processed: "
    f"{len(video_data)} / {len(video_files)}"
)
print()


# ============================================================
# SAVE SAMPLE FRAMES
# ============================================================

print("Saving representative frames...")

for filename, data in video_data.items():

    video_index = get_video_index(
        Path(filename)
    )

    video_frame_dir = (
        FRAMES_DIR /
        f"video_{video_index}"
    )

    video_frame_dir.mkdir(
        exist_ok=True
    )

    for i, frame in enumerate(
        data["frames"]
    ):

        output = (
            video_frame_dir /
            f"sample_{i:02d}.jpg"
        )

        cv2.imwrite(
            str(output),
            frame
        )


# ============================================================
# VIDEO-TO-VIDEO PHASH DISTANCE
# ============================================================

def best_phash_alignment(
    data_a,
    data_b
):
    """
    Compare sampled frames while allowing
    temporal / rotational phase shifts.

    Instead of:

        frame 1 ↔ frame 1

    we test several circular offsets:

        frame 1 ↔ frame 2
        frame 2 ↔ frame 3
        ...

    This matters because two rotations may start
    at different angular positions.
    """

    hashes_a = data_a["hashes"]
    hashes_b = data_b["hashes"]

    n = min(
        len(hashes_a),
        len(hashes_b)
    )

    if n == 0:
        return {
            "mean_distance": None,
            "best_distance": None,
            "best_offset": None
        }

    best_mean = float("inf")
    best_offset = None

    # Test every circular offset
    for offset in range(n):

        distances = []

        for i in range(n):

            j = (
                i + offset
            ) % n

            distances.append(
                hamming_distance(
                    hashes_a[i],
                    hashes_b[j]
                )
            )

        mean_distance = np.mean(
            distances
        )

        if mean_distance < best_mean:

            best_mean = mean_distance
            best_offset = offset

    return {
        "mean_distance": float(
            best_mean
        ),
        "best_distance": float(
            best_mean
        ),
        "best_offset": int(
            best_offset
        )
    }


# ============================================================
# FRAME SIMILARITY WITH ALIGNMENT
# ============================================================

def aligned_pixel_similarity(
    data_a,
    data_b,
    offset
):

    frames_a = data_a["frames"]
    frames_b = data_b["frames"]

    n = min(
        len(frames_a),
        len(frames_b)
    )

    if n == 0:
        return None

    scores = []

    for i in range(n):

        j = (
            i + offset
        ) % n

        score = pixel_similarity(
            frames_a[i],
            frames_b[j]
        )

        scores.append(
            score
        )

    return {
        "mean": float(
            np.mean(scores)
        ),
        "median": float(
            np.median(scores)
        ),
        "minimum": float(
            np.min(scores)
        ),
        "p10": float(
            np.percentile(
                scores,
                10
            )
        )
    }


# ============================================================
# VIDEO COMPARISON
# ============================================================

def compare_videos(
    filename_a,
    filename_b
):

    data_a = video_data[
        filename_a
    ]

    data_b = video_data[
        filename_b
    ]

    # --------------------------------------------
    # Stage 1: perceptual hash
    # --------------------------------------------

    phash_result = (
        best_phash_alignment(
            data_a,
            data_b
        )
    )

    distance = (
        phash_result["best_distance"]
    )

    if distance is None:

        return None

    # Convert Hamming distance to rough similarity.
    # 256 bits for hash_size=16.
    phash_similarity = (
        1.0 - (
            distance / 256.0
        )
    )

    # --------------------------------------------
    # Stage 2:
    # Pixel comparison
    # --------------------------------------------

    pixel_result = (
        aligned_pixel_similarity(
            data_a,
            data_b,
            phash_result["best_offset"]
        )
    )

    if pixel_result is None:
        return None

    # --------------------------------------------
    # Combined score
    # --------------------------------------------

    # Pixel similarity is deliberately weighted
    # more heavily than pHash.
    combined = (
        0.35 * phash_similarity
        +
        0.65 * pixel_result["mean"]
    )

    return {

        "video_a": filename_a,
        "video_b": filename_b,

        "frames_a": data_a[
            "frame_count"
        ],

        "frames_b": data_b[
            "frame_count"
        ],

        "duration_a": data_a[
            "duration"
        ],

        "duration_b": data_b[
            "duration"
        ],

        "phash_distance": distance,

        "phash_similarity": phash_similarity,

        "best_offset": phash_result[
            "best_offset"
        ],

        "pixel_similarity_mean": pixel_result[
            "mean"
        ],

        "pixel_similarity_median": pixel_result[
            "median"
        ],

        "pixel_similarity_min": pixel_result[
            "minimum"
        ],

        "pixel_similarity_p10": pixel_result[
            "p10"
        ],

        "combined_similarity": combined
    }


# ============================================================
# PAIRWISE COMPARISON
# ============================================================

pairs = list(
    combinations(
        video_data.keys(),
        2
    )
)

print()
print("=" * 100)
print("PAIRWISE VIDEO SIMILARITY")
print("=" * 100)
print()

print(
    f"Videos: {len(video_data)}"
)

print(
    f"Video pairs: {len(pairs):,}"
)

print()

results = []

for filename_a, filename_b in tqdm(
    pairs,
    desc="Comparing pairs"
):

    try:

        result = compare_videos(
            filename_a,
            filename_b
        )

        if result is not None:
            results.append(result)

    except Exception as e:

        print(
            f"\nComparison error: "
            f"{filename_a} vs "
            f"{filename_b}: {e}"
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY SIMILARITY
# ============================================================

results_df = results_df.sort_values(
    "combined_similarity",
    ascending=False
)


# ============================================================
# SAVE ALL PAIRS
# ============================================================

all_pairs_csv = (
    OUTPUT_DIR /
    "all_video_pairs.csv"
)

results_df.to_csv(
    all_pairs_csv,
    index=False
)


# ============================================================
# HIGH SIMILARITY PAIRS
# ============================================================

high_similarity = results_df[
    results_df[
        "combined_similarity"
    ]
    >= HIGH_SIMILARITY_THRESHOLD
].copy()


very_high_similarity = results_df[
    results_df[
        "combined_similarity"
    ]
    >= VERY_HIGH_SIMILARITY_THRESHOLD
].copy()


high_csv = (
    OUTPUT_DIR /
    "high_similarity_pairs.csv"
)

very_high_csv = (
    OUTPUT_DIR /
    "very_high_similarity_pairs.csv"
)

high_similarity.to_csv(
    high_csv,
    index=False
)

very_high_similarity.to_csv(
    very_high_csv,
    index=False
)


# ============================================================
# PRINT TOP PAIRS
# ============================================================

print()
print("=" * 100)
print("TOP 30 MOST SIMILAR VIDEO PAIRS")
print("=" * 100)
print()

columns = [
    "video_a",
    "video_b",
    "phash_distance",
    "phash_similarity",
    "pixel_similarity_mean",
    "combined_similarity",
    "best_offset"
]

top = results_df.head(30)

for _, row in top.iterrows():

    print(
        f"{row['video_a']:<15} ↔ "
        f"{row['video_b']:<15} "
        f"pHash={row['phash_distance']:>5.1f} "
        f"pSim={row['phash_similarity']:.3f} "
        f"pixel={row['pixel_similarity_mean']:.3f} "
        f"FINAL={row['combined_similarity']:.3f} "
        f"offset={row['best_offset']}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("SIMILARITY SUMMARY")
print("=" * 100)
print()

print(
    f"Total video pairs analyzed : "
    f"{len(results_df):,}"
)

print(
    f"Pairs >= 90% similarity     : "
    f"{len(high_similarity)}"
)

print(
    f"Pairs >= 95% similarity     : "
    f"{len(very_high_similarity)}"
)

if len(results_df) > 0:

    print()

    print(
        f"Highest similarity         : "
        f"{results_df['combined_similarity'].max():.4f}"
    )

    print(
        f"Median pair similarity     : "
        f"{results_df['combined_similarity'].median():.4f}"
    )

    print(
        f"Lowest pair similarity     : "
        f"{results_df['combined_similarity'].min():.4f}"
    )


# ============================================================
# CREATE NEAR-DUPLICATE GROUPS
# ============================================================

print()
print("=" * 100)
print("NEAR-DUPLICATE GROUPS")
print("=" * 100)
print()

# Graph-based grouping.
# If A is very similar to B and B is very similar to C,
# they belong to the same connected group.

parent = {
    filename: filename
    for filename in video_data.keys()
}


def find(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


for _, row in high_similarity.iterrows():

    union(
        row["video_a"],
        row["video_b"]
    )


groups = {}

for filename in video_data.keys():

    root = find(filename)

    groups.setdefault(
        root,
        []
    ).append(filename)


groups = [
    group
    for group in groups.values()
    if len(group) > 1
]

groups.sort(
    key=len,
    reverse=True
)


if not groups:

    print(
        "No near-duplicate groups found "
        f"at >= {HIGH_SIMILARITY_THRESHOLD:.0%}."
    )

else:

    for i, group in enumerate(
        groups,
        start=1
    ):

        print()
        print(
            f"Group {i} "
            f"({len(group)} videos):"
        )

        for filename in sorted(
            group,
            key=lambda x: get_video_index(
                Path(x)
            )
        ):

            print(
                f"  {filename}"
            )


# ============================================================
# SAVE GROUPS
# ============================================================

group_records = []

for group_id, group in enumerate(
    groups,
    start=1
):

    for filename in group:

        group_records.append({
            "group_id": group_id,
            "group_size": len(group),
            "filename": filename
        })


groups_csv = (
    OUTPUT_DIR /
    "near_duplicate_groups.csv"
)

pd.DataFrame(
    group_records
).to_csv(
    groups_csv,
    index=False
)


# ============================================================
# CONTACT SHEET GENERATOR
# ============================================================

def create_contact_sheet(
    filename_a,
    filename_b,
    offset,
    output_path
):

    data_a = video_data[
        filename_a
    ]

    data_b = video_data[
        filename_b
    ]

    frames_a = data_a["frames"]
    frames_b = data_b["frames"]

    n = min(
        len(frames_a),
        len(frames_b)
    )

    if n == 0:
        return

    # Pick 6 representative positions.
    positions = np.linspace(
        0,
        n - 1,
        6,
        dtype=int
    )

    rows = []

    for i in positions:

        j = (
            i + offset
        ) % n

        a = cv2.resize(
            frames_a[i],
            (
                320,
                428
            )
        )

        b = cv2.resize(
            frames_b[j],
            (
                320,
                428
            )
        )

        # Labels
        cv2.putText(
            a,
            f"A frame {i}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            b,
            f"B frame {j}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        combined = np.hstack(
            [a, b]
        )

        rows.append(
            combined
        )

    sheet = np.vstack(
        rows
    )

    cv2.imwrite(
        str(output_path),
        sheet
    )


# ============================================================
# SAVE CONTACT SHEETS FOR TOP PAIRS
# ============================================================

print()
print(
    "Creating contact sheets for top "
    "similar pairs..."
)

TOP_CONTACT_SHEETS = min(
    20,
    len(results_df)
)

for rank in range(
    TOP_CONTACT_SHEETS
):

    row = results_df.iloc[
        rank
    ]

    output = (
        SUSPICIOUS_DIR /
        (
            f"rank_{rank + 1:02d}_"
            f"{row['video_a'].replace('.mp4', '')}_"
            f"{row['video_b'].replace('.mp4', '')}.jpg"
        )
    )

    create_contact_sheet(
        row["video_a"],
        row["video_b"],
        int(row["best_offset"]),
        output
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)
print()

print(
    f"All pair comparisons:"
)
print(
    f"  {all_pairs_csv}"
)

print()

print(
    f">=90% similarity:"
)
print(
    f"  {high_csv}"
)

print()

print(
    f">=95% similarity:"
)
print(
    f"  {very_high_csv}"
)

print()

print(
    f"Near-duplicate groups:"
)
print(
    f"  {groups_csv}"
)

print()

print(
    f"Contact sheets:"
)
print(
    f"  {SUSPICIOUS_DIR}"
)

print()
print("=" * 100)
print("SIMILARITY ANALYSIS COMPLETE")
print("=" * 100)
print()
