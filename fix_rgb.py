from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import multiprocessing as mp

SRC = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/"
    "notebooks/rasberry/trail__videos/final_video"
)

DST = Path(
    "/home/gautamw7/Desktop/Projects/Autonex/Paint_Box/"
    "notebooks/rasberry/trail__videos/final_video_rgb"
)

WORKERS = max(1, mp.cpu_count() - 1)


def fix_image(args):
    src, dst = args

    src = Path(src)
    dst = Path(dst)

    img = cv2.imread(
        str(src),
        cv2.IMREAD_COLOR
    )

    if img is None:
        return False, str(src)

    # Swap RED <-> BLUE channels.
    #
    # cv2.imread() gives BGR.
    # The saved dataset has the R/B channels reversed.
    #
    # BGR -> RGB
    corrected = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # IMPORTANT:
    # cv2.imwrite() expects BGR, so convert back before writing.
    corrected_bgr = cv2.cvtColor(
        corrected,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(dst),
        corrected_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    return True, str(src)


def main():

    images = list(
        SRC.rglob("*.jpg")
    )

    print("=" * 70)
    print("RGB CHANNEL CORRECTION")
    print("=" * 70)
    print(f"Images found : {len(images)}")
    print(f"Workers      : {WORKERS}")
    print(f"Source       : {SRC}")
    print(f"Output       : {DST}")
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
                fix_image,
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

            if i % 100 == 0 or i == len(futures):
                print(
                    f"[{i}/{len(futures)}] "
                    f"success={success} "
                    f"failed={failed}"
                )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Images processed : {success}")
    print(f"Failed           : {failed}")
    print()
    print("Corrected dataset:")
    print(DST)
    print("=" * 70)


if __name__ == "__main__":
    main()
