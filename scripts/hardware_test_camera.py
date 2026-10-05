from picamera2 import Picamera2
from PIL import Image
import cv2

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1456, 1088),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

frame = picam2.capture_array()

print("Shape:", frame.shape)
print("Dtype:", frame.dtype)

# 1. Save directly
Image.fromarray(frame).save(
    "test_no_conversion.jpg",
    quality=95
)

# 2. Swap BGR -> RGB
converted = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)

Image.fromarray(converted).save(
    "test_bgr_to_rgb.jpg",
    quality=95
)

picam2.stop()

print()
print("Saved:")
print("  test_no_conversion.jpg")
print("  test_bgr_to_rgb.jpg")
print()
print("Open both images and compare the can colors.")
