from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import multiprocessing as mp


SRC = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/"
    "notebooks/rasberry/trail__videos/final_video"
)

DST = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/"
    "notebooks/rasberry/trail__videos/final_video_corrected2"
)

WORKERS = max(1, mp.cpu_count() - 1)


# ============================================================
# TUNING
# ============================================================

# IMPORTANT:
# output = input ** GAMMA
#
# < 1.0 = brighter
#
GAMMA = 0.45

# Limit white balance correction
MAX_GAIN = 1.5

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_GRID = 8

JPEG_QUALITY = 95


# ============================================================
# GAMMA
# ============================================================

def gamma_correct(img, gamma):

    table = np.array([
        ((i / 255.0) ** gamma) * 255.0
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(img, table)


# ============================================================
# GRAY WORLD WHITE BALANCE
# ============================================================

def white_balance(img):

    # BGR
    b, g, r = cv2.split(
        img.astype(np.float32)
    )

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)

    mean_gray = (
        mean_b +
        mean_g +
        mean_r
    ) / 3.0

    # Calculate gains
    gain_b = mean_gray / max(mean_b, 1e-6)
    gain_g = mean_gray / max(mean_g, 1e-6)
    gain_r = mean_gray / max(mean_r, 1e-6)

    # Prevent extreme correction
    gain_b = np.clip(
        gain_b,
        1.0 / MAX_GAIN,
        MAX_GAIN
    )

    gain_g = np.clip(
        gain_g,
        1.0 / MAX_GAIN,
        MAX_GAIN
    )

    gain_r = np.clip(
        gain_r,
        1.0 / MAX_GAIN,
        MAX_GAIN
    )

    b *= gain_b
    g *= gain_g
    r *= gain_r

    corrected = cv2.merge([
        np.clip(b, 0, 255),
        np.clip(g, 0, 255),
        np.clip(r, 0, 255)
    ]).astype(np.uint8)

    return corrected


# ============================================================
# LOCAL CONTRAST
# ============================================================

def clahe_correct(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(
            CLAHE_GRID,
            CLAHE_GRID
        )
    )

    l = clahe.apply(l)

    lab = cv2.merge([
        l,
        a,
        b
    ])

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(args):

    src, dst = args

    src = Path(src)
    dst = Path(dst)

    img = cv2.imread(
        str(src),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return False, str(src)

    # --------------------------------------------------------
    # 1. CORRECT GAMMA
    # --------------------------------------------------------

    img = gamma_correct(
        img,
        GAMMA
    )

    # --------------------------------------------------------
    # 2. WHITE BALANCE
    # --------------------------------------------------------

    img = white_balance(
        img
    )

    # --------------------------------------------------------
    # 3. LOCAL CONTRAST
    # --------------------------------------------------------

    img = clahe_correct(
        img
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ok = cv2.imwrite(
        str(dst),
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]
    )

    return ok, str(src)


# ============================================================
# MAIN
# ============================================================

def main():

    images = list(
        SRC.rglob("*.jpg")
    )

    print("=" * 70)
    print("FRAME CORRECTION V2")
    print("=" * 70)

    print(f"Images       : {len(images)}")
    print(f"Workers      : {WORKERS}")
    print()
    print("Gamma        :", GAMMA)
    print("Max WB gain  :", MAX_GAIN)
    print("CLAHE clip   :", CLAHE_CLIP)
    print()
    print("Source:")
    print(SRC)
    print()
    print("Output:")
    print(DST)

    print("=" * 70)

    tasks = []

    for src in images:

        relative = src.relative_to(SRC)

        dst = DST / relative

        tasks.append(
            (
                src,
                dst
            )
        )

    success = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_image,
                task
            )
            for task in tasks
        ]

        for i, future in enumerate(
            as_completed(futures),
            start=1
        ):

            ok, path = future.result()

            if ok:
                success += 1
            else:
                failed += 1

            if (
                i % 100 == 0
                or i == len(futures)
            ):
                print(
                    f"[{i}/{len(futures)}] "
                    f"success={success} "
                    f"failed={failed}"
                )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print("Processed :", success)
    print("Failed    :", failed)
    print()
    print(DST)
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
PY

