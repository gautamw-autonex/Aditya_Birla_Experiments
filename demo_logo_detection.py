"""Demo script for logo detection system.

Shows the full pipeline:
1. Curvature augmentation of LogoDet-3K
2. Evaluation with text-based metrics
3. Prediction on test images
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paint_box.logo_detection.curvature_augment import CylindricalWarp, CylindricalWarpConfig
from paint_box.logo_detection.evaluator import LogoEvaluator
from paint_box.logo_detection.data_loader import LogoDataLoader
from paint_box.logo_detection.config import CLASS_NAMES, CURVATURE_PARAMS, DETECT_CONFIG, PRETRAIN_WEIGHTS


def demo_augmentation():
    """Demonstrate curvature augmentation on LogoDet-3K."""
    print("\n" + "=" * 70)
    print("  1. CURVATURE AUGMENTATION DEMO")
    print("=" * 70)

    warp = CylindricalWarp()
    print(f"\n  Cylindrical Warping Parameters:")
    print(f"    0.5L can radius: {CURVATURE_PARAMS['radius_05l']}px (tight curve)")
    print(f"    1L can radius: {CURVATURE_PARAMS['radius_1l']}px (medium curve)")
    print(f"    4L can radius: {CURVATURE_PARAMS['radius_4l']}px (flatter curve)")
    print(f"    View angle range: {CURVATURE_PARAMS['view_angle_min']}° to {CURVATURE_PARAMS['view_angle_max']}°")
    print(f"    Distortion strength: {CURVATURE_PARAMS['distortion_strength_min']} to {CURVATURE_PARAMS['distortion_strength_max']}")

    # Test with a sample image
    test_img_path = Path("data/logo/logoDet3k_test/Clothes/1.jpg")
    if test_img_path.exists():
        img = cv2.imread(str(test_img_path))
        print(f"\n  Test image: {test_img_path}")
        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")

        for cat, radius in [("0.5L can", CURVATURE_PARAMS['radius_05l']),
                             ("1L can", CURVATURE_PARAMS['radius_1l']),
                             ("4L can", CURVATURE_PARAMS['radius_4l'])]:
            warped = warp.warp_to_cylinder(img, radius=radius, view_angle=15.0, distortion_strength=0.7)
            print(f"  {cat} warped size: {warped.shape[1]}x{warped.shape[0]}")
            out_path = f"output/logo_detection/demo_warp_{cat.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, warped)
            print(f"    Saved to: {out_path}")
    else:
        print(f"\n  Test image not found at {test_img_path}")
        print(f"  Creating a dummy test image...")
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for cat, radius in [("0.5L can", CURVATURE_PARAMS['radius_05l']),
                             ("1L can", CURVATURE_PARAMS['radius_1l']),
                             ("4L can", CURVATURE_PARAMS['radius_4l'])]:
            warped = warp.warp_to_cylinder(dummy, radius=radius)
            print(f"  {cat} warped: {warped.shape[1]}x{warped.shape[0]}")
    print("  [DONE] Curvature augmentation demo complete")


def demo_evaluation():
    """Demonstrate text-based evaluation."""
    print("\n" + "=" * 70)
    print("  2. EVALUATION METRICS DEMO (No Image Viewing)")
    print("=" * 70)

    evaluator = LogoEvaluator()

    # Simulate predictions for 5 test images
    np.random.seed(42)
    predictions = []
    ground_truths = []

    for i in range(5):
        # Simulate varying numbers of detections per image
        num_det = np.random.randint(0, 4)
        boxes = []
        scores = []
        class_ids = []

        for _ in range(num_det):
            x1 = np.random.uniform(0, 0.8)
            y1 = np.random.uniform(0, 0.8)
            w = np.random.uniform(0.05, 0.2)
            h = np.random.uniform(0.05, 0.2)
            boxes.append([x1 * 640, y1 * 640, (x1 + w) * 640, (y1 + h) * 640])
            scores.append(np.random.uniform(0.3, 0.95))
            class_ids.append(np.random.randint(0, 3))

        predictions.append({
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "num_detections": len(boxes),
        })

        # Ground truth: 1-2 logos per image
        num_gt = np.random.randint(1, 3)
        gt = []
        for _ in range(num_gt):
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            bw = np.random.uniform(0.05, 0.15)
            bh = np.random.uniform(0.05, 0.15)
            gt.append((np.random.randint(0, 3), cx, cy, bw, bh))
        ground_truths.append(gt)

    metrics = evaluator.evaluate_predictions(predictions, ground_truths)

    # Print the full evaluation report
    report = evaluator.print_evaluation_report(metrics)
    print(report)

    return metrics


def demo_data_loader():
    """Demonstrate data loading."""
    print("\n" + "=" * 70)
    print("  3. DATA LOADER DEMO")
    print("=" * 70)

    loader = LogoDataLoader()

    # Show LogoDet-3K subsets
    subsets = loader.get_det3k_subsets()
    print(f"\n  LogoDet-3K categories available: {len(subsets)}")
    for cat, imgs in sorted(subsets.items())[:10]:
        print(f"    {cat}: {len(imgs)} images")

    # Show data structure
    print(f"\n  Expected data structure:")
    print(f"    data/logo/train/images/ + labels/")
    print(f"    data/logo/val/images/ + labels/")
    print(f"    data/logo/test/images/ + labels/")
    print(f"    data/logo/augmented_logoDet3k/05l|1l|4l/")

    # Check test data
    test_imgs = list((loader.data_dir / "test" / "images").glob("*"))
    test_lbls = list((loader.data_dir / "test" / "labels").glob("*.txt"))
    print(f"\n  Test data: {len(test_imgs)} images, {len(test_lbls)} labels")
    print("  [DONE] Data loader demo complete")


def main():
    print("\n" + "=" * 70)
    print("  PAINT BOX - LOGO DETECTION DEMO")
    print("  PaddlePaddle YOLOE-S | CPU Inference | No Image Viewing")
    print("=" * 70)

    demo_data_loader()
    demo_augmentation()
    metrics = demo_evaluation()

    print("\n" + "=" * 70)
    print("  HOW TO USE (CLI)")
    print("=" * 70)
    print("""
  # Augment LogoDet-3K with cylindrical warping:
  poetry run python -m paint_box.logo_detection.cli augment \\
    --det3k-dir data/logo/LogoDet-3K \\
    --output-dir data/logo/augmented_logoDet3k \\
    --sizes 05l 1l 4l

  # Evaluate model on test data:
  poetry run python -m paint_box.logo_detection.cli eval \\
    --test-dir data/logo/test \\
    --output-dir output/logo_detection

  # Predict on a single image:
  poetry run python -m paint_box.logo_detection.cli predict \\
    data/test_qr_1.png \\
    --conf-thresh 0.25

  # Quick evaluation (uses augmented data):
  poetry run python -m paint_box.logo_detection.cli quick-eval \\
    --data-dir data/logo

  # Run all tests:
  poetry run pytest tests/ -v
    """)

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
