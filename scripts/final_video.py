from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shutil


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/notebooks/rasberry"
)

VIDEO_DIR = BASE_DIR / "trail__videos"

# Similarity output from your previous analysis
SIMILARITY_CSV = (
    VIDEO_DIR
    / "similarity_analysis"
    / "all_video_pairs.csv"
)

FINAL_DIR = VIDEO_DIR / "final_video"

# Current conservative threshold
SIMILARITY_THRESHOLD = 0.90

# Extract every 2nd frame
STRIDE = 2

# JPEG quality
JPEG_QUALITY = 95

# Number of parallel workers
MAX_WORKERS = max(1, mp.cpu_count() - 1)


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        result = {}

        for item in self.parent:
            root = self.find(item)
            result.setdefault(root, []).append(item)

        return list(result.values())


# ============================================================
# FIND ALL VIDEOS
# ============================================================

def get_videos():

    videos = sorted(
        VIDEO_DIR.glob("video_*.mp4"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    return videos


# ============================================================
# BUILD SIMILARITY GROUPS
# ============================================================

def build_unique_groups(videos):

    video_names = [v.stem for v in videos]

    uf = UnionFind(video_names)

    if not SIMILARITY_CSV.exists():

        raise FileNotFoundError(
            f"Similarity file not found:\n{SIMILARITY_CSV}"
        )

    df = pd.read_csv(SIMILARITY_CSV)

    print(f"Loaded {len(df):,} video-pair comparisons")

    # --------------------------------------------------------
    # Find the similarity column
    # --------------------------------------------------------

    similarity_candidates = [
        "combined_similarity",
	"final_similarity",
        "final",
        "similarity",
        "FINAL"
    ]

    similarity_col = None

    for col in similarity_candidates:
        if col in df.columns:
            similarity_col = col
            break

    if similarity_col is None:

        raise ValueError(
            f"Could not find similarity column.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find video columns
    # --------------------------------------------------------

    video_a_candidates = [
        "video_a",
        "video1",
        "video_1",
        "videoA"
    ]

    video_b_candidates = [
        "video_b",
        "video2",
        "video_2",
        "videoB"
    ]

    col_a = next(
        (c for c in video_a_candidates if c in df.columns),
        None
    )

    col_b = next(
        (c for c in video_b_candidates if c in df.columns),
        None
    )

    if col_a is None or col_b is None:

        raise ValueError(
            "Could not identify video pair columns.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Group similar videos
    # --------------------------------------------------------

    similar_count = 0

    for _, row in df.iterrows():

        a = str(row[col_a]).replace(".mp4", "")
        b = str(row[col_b]).replace(".mp4", "")

        similarity = float(row[similarity_col])

        if (
            similarity >= SIMILARITY_THRESHOLD
            and a in uf.parent
            and b in uf.parent
        ):

            uf.union(a, b)
            similar_count += 1

    groups = uf.groups()

    # --------------------------------------------------------
    # Sort groups by video number
    # --------------------------------------------------------

    def video_number(name):
        return int(name.split("_")[1])

    groups = [
        sorted(group, key=video_number)
        for group in groups
    ]

    groups.sort(
        key=lambda g: video_number(g[0])
    )

    print()
    print("=" * 70)
    print("SIMILARITY GROUPING")
    print("=" * 70)
    print(f"Similarity threshold : {SIMILARITY_THRESHOLD}")
    print(f"Similar pairs        : {similar_count}")
    print(f"Original videos      : {len(videos)}")
    print(f"Unique groups        : {len(groups)}")
    print("=" * 70)

    return groups


# ============================================================
# CHOOSE REPRESENTATIVE
# ============================================================

def choose_representatives(groups):

    representatives = []

    for group in groups:

        # Current strategy:
        # choose the first/lowest-numbered video.
        representative = group[0]

        representatives.append({
            "representative": representative,
            "group": group,
            "group_size": len(group)
        })

    return representatives


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(args):

    video_path, output_dir = args

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        return {
            "video": video_path.stem,
            "status": "ERROR",
            "frames": 0,
            "message": "Could not open video"
        }

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_index = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # STRIDE
        # ----------------------------------------------------

        if frame_index % STRIDE != 0:

            frame_index += 1
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # OpenCV gives us BGR.
        #
        # Convert:
        #
        # BGR -> RGB
        #
        # Then PIL saves the image with correct RGB channels.
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        output_file = (
            output_dir
            / f"frame_{frame_index:06d}.jpg"
        )

        image.save(
            output_file,
            "JPEG",
            quality=JPEG_QUALITY,
            subsampling=0
        )

        saved += 1
        frame_index += 1

    cap.release()

    return {
        "video": video_path.stem,
        "status": "OK",
        "frames_original": total_frames,
        "frames_saved": saved,
        "fps": fps,
        "width": width,
        "height": height,
        "message": ""
    }


# ============================================================
# MAIN
# ============================================================

def main():

    videos = get_videos()

    if not videos:

        raise RuntimeError(
            f"No videos found in {VIDEO_DIR}"
        )

    print()
    print("=" * 70)
    print("FINAL DATASET CREATION")
    print("=" * 70)

    print(f"Videos found : {len(videos)}")
    print(f"Stride       : {STRIDE}")
    print(f"Workers      : {MAX_WORKERS}")

    # --------------------------------------------------------
    # BUILD UNIQUE GROUPS
    # --------------------------------------------------------

    groups = build_unique_groups(videos)

    representatives = choose_representatives(groups)

    # --------------------------------------------------------
    # CLEAN FINAL DIRECTORY
    # --------------------------------------------------------

    if FINAL_DIR.exists():

        print()
        print(f"Removing existing:\n{FINAL_DIR}")

        shutil.rmtree(FINAL_DIR)

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE GROUP INFORMATION
    # --------------------------------------------------------

    group_rows = []

    for idx, item in enumerate(representatives, start=1):

        representative = item["representative"]

        group_rows.append({
            "group_id": idx,
            "representative": representative,
            "group_size": item["group_size"],
            "members": ",".join(item["group"])
        })

    group_df = pd.DataFrame(group_rows)

    group_csv = (
        FINAL_DIR
        / "unique_video_groups.csv"
    )

    group_df.to_csv(
        group_csv,
        index=False
    )

    # --------------------------------------------------------
    # BUILD PROCESSING TASKS
    # --------------------------------------------------------

    video_lookup = {
        v.stem: v
        for v in videos
    }

    tasks = []

    for item in representatives:

        video_name = item["representative"]

        source = video_lookup[video_name]

        output = FINAL_DIR / video_name

        tasks.append(
            (
                str(source),
                str(output)
            )
        )

    # --------------------------------------------------------
    # PARALLEL PROCESSING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXTRACTING FRAMES")
    print("=" * 70)

    results = []

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_video,
                task
            )
            for task in tasks
        ]

        for i, future in enumerate(
            as_completed(futures),
            start=1
        ):

            result = future.result()

            results.append(result)

            print(
                f"[{i:02d}/{len(futures)}] "
                f"{result['video']} -> "
                f"{result['frames_saved']} frames "
                f"[{result['status']}]"
            )

    # --------------------------------------------------------
    # SAVE PROCESSING SUMMARY
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "video"
    )

    result_csv = (
        FINAL_DIR
        / "dataset_summary.csv"
    )

    result_df.to_csv(
        result_csv,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    total_saved = int(
        result_df["frames_saved"].sum()
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Original videos : {len(videos)}")
    print(f"Unique videos   : {len(representatives)}")
    print(f"Frames saved    : {total_saved}")
    print(f"Stride          : {STRIDE}")
    print()
    print(f"Dataset:")
    print(FINAL_DIR)
    print()
    print(f"Groups CSV:")
    print(group_csv)
    print()
    print(f"Summary CSV:")
    print(result_csv)
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
