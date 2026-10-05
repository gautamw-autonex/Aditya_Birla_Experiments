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
    "notebooks/rasberry/trail__videos/final_video_corrected"
)

WORKERS = max(1, mp.cpu_count() - 1)

# ------------------------------------------------------------
# TUNING
# ------------------------------------------------------------

# Gamma < 1 makes dark images brighter.
GAMMA = 0.65

# Manual colour gains.
#
# The image has a blue/cyan cast, so:
#   RED   -> increase
#   GREEN -> slight increase
#   BLUE  -> decrease
#
RED_GAIN   = 1.18
GREEN_GAIN = 1.03
BLUE_GAIN  = 0.82

# Contrast
CONTRAST = 1.08

# Brightness offset
BRIGHTNESS = 3


def gamma_correct(img, gamma):

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(img, table)


def colour_correct(img):

    # OpenCV image is BGR.
    b, g, r = cv2.split(img)

    b = np.clip(
        b.astype(np.float32) * BLUE_GAIN,
        0,
        255
    ).astype(np.uint8)

    g = np.clip(
        g.astype(np.float32) * GREEN_GAIN,
        0,
        255
    ).astype(np.uint8)

    r = np.clip(
        r.astype(np.float32) * RED_GAIN,
        0,
        255
    ).astype(np.uint8)

    return cv2.merge([b, g, r])


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
    # 1. GAMMA / EXPOSURE
    # --------------------------------------------------------

    img = gamma_correct(
        img,
        GAMMA
    )

    # --------------------------------------------------------
    # 2. COLOUR CAST CORRECTION
    # --------------------------------------------------------

    img = colour_correct(img)

    # --------------------------------------------------------
    # 3. CONTRAST
    # --------------------------------------------------------

    img = cv2.convertScaleAbs(
        img,
        alpha=CONTRAST,
        beta=BRIGHTNESS
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(dst),
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    return True, str(src)


def main():

    images = list(
        SRC.rglob("*.jpg")
    )

    print("=" * 70)
    print("SAVED FRAME COLOUR / EXPOSURE CORRECTION")
    print("=" * 70)

    print(f"Images  : {len(images)}")
    print(f"Workers : {WORKERS}")
    print()
    print("Settings:")
    print(f"Gamma       : {GAMMA}")
    print(f"Red gain    : {RED_GAIN}")
    print(f"Green gain  : {GREEN_GAIN}")
    print(f"Blue gain   : {BLUE_GAIN}")
    print(f"Contrast    : {CONTRAST}")
    print(f"Brightness  : {BRIGHTNESS}")

    print()
    print(f"Output:")
    print(DST)

    print("=" * 70)

    tasks = []

    for src in images:

        relative = src.relative_to(SRC)

        dst = DST / relative

        tasks.append(
            (src, dst)
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

    print(f"Processed : {success}")
    print(f"Failed    : {failed}")
    print()
    print("Corrected dataset:")
    print(DST)
    print("=" * 70)


if __name__ == "__main__":
    mp.freeze_support()
    main()
