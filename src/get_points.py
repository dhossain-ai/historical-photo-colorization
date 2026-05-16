import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "input" / "image1.jpg"

points = []

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point added: [{x}, {y}]")

        img_copy = image.copy()

        # Draw clicked points
        for point in points:
            cv2.circle(img_copy, tuple(point), 5, (0, 0, 255), -1)

        # Draw connecting lines
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(img_copy, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)

        cv2.imshow("Click points - press C to clear, Q to quit", img_copy)


image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

display_width = 1000
scale = display_width / image.shape[1]
display_height = int(image.shape[0] * scale)

image = cv2.resize(image, (display_width, display_height))

cv2.imshow("Click points - press C to clear, Q to quit", image)
cv2.setMouseCallback("Click points - press C to clear, Q to quit", mouse_callback)

print("Click around one region.")
print("Press Q when done.")
print("Press C to clear points.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        points = []
        cv2.imshow("Click points - press C to clear, Q to quit", image)
        print("Cleared points.")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

print("\nFinal points:")
print(points)

print("\nCopy this into your colorization code:")
print("region_points = [")
for p in points:
    print(f"    {p},")
print("]")