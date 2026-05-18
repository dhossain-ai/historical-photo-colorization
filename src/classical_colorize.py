from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_IMAGE = PROJECT_ROOT / "input" / "image1.jpg"
OUTPUT_IMAGE = PROJECT_ROOT / "output" / "classical_colorized.png"

DISPLAY_WIDTH = 1000


# -----------------------------
# Manually selected mask points
# Image was resized to width 1000
# -----------------------------

SKY_POINTS = [
    [570, 178], [747, 182], [932, 188], [937, 171], [935, 162],
    [927, 153], [927, 137], [923, 118], [923, 89], [929, 69],
    [942, 58], [961, 43], [978, 35], [995, 30], [993, 5],
    [969, 3], [888, 2], [722, 0], [645, 0], [617, 0],
    [600, 28], [583, 46], [572, 74], [573, 101], [576, 124],
    [574, 143], [571, 174],
]

SEA_POINTS = [
    [578, 232], [577, 206], [567, 186], [638, 188], [746, 191],
    [837, 195], [919, 198], [947, 202], [951, 214], [960, 227],
    [987, 223], [995, 226], [997, 262], [999, 287], [997, 306],
    [997, 382], [998, 404], [862, 373], [763, 348], [715, 342],
    [700, 321], [687, 302], [687, 283], [688, 256], [690, 227],
    [679, 220], [671, 211], [656, 203], [630, 200], [615, 205],
    [605, 212], [591, 224], [579, 236],
]

BOAT_POINTS = [
    [4, 366], [37, 359], [61, 354], [82, 344], [108, 324],
    [113, 295], [110, 272], [116, 242], [127, 226], [155, 200],
    [185, 186], [177, 172], [175, 151], [188, 135], [203, 144],
    [215, 159], [227, 163], [246, 170], [277, 168], [294, 172],
    [307, 160], [316, 148], [333, 134], [337, 117], [328, 105],
    [319, 80], [315, 64], [312, 38], [326, 36], [341, 20],
    [353, 20], [371, 22], [387, 24], [398, 31], [412, 32],
    [431, 48], [419, 62], [423, 82], [417, 102], [413, 116],
    [429, 127], [449, 134], [463, 148], [477, 168], [482, 184],
    [485, 194], [495, 214], [504, 234], [505, 248], [501, 274],
    [493, 291], [477, 304], [469, 315], [468, 326], [468, 348],
    [477, 399], [483, 430], [484, 448], [485, 476], [495, 496],
    [504, 531], [510, 548], [527, 540], [552, 519], [565, 506],
    [561, 486], [559, 474], [549, 445], [547, 411], [543, 376],
    [548, 358], [559, 344], [575, 333], [577, 328], [576, 310],
    [576, 288], [583, 276], [573, 258], [573, 242], [585, 222],
    [611, 205], [636, 198], [650, 204], [675, 213], [680, 233],
    [688, 258], [688, 274], [685, 296], [685, 310], [694, 321],
    [711, 338], [717, 350], [731, 374], [745, 398], [759, 407],
    [776, 419], [803, 424], [827, 427], [846, 421], [876, 416],
    [896, 402], [911, 406], [913, 386], [874, 379], [835, 379],
    [817, 370], [790, 358], [771, 354], [742, 347], [723, 338],
    [711, 325], [701, 314], [697, 303], [700, 280], [696, 251],
    [685, 213], [663, 201], [640, 187], [616, 197], [597, 204],
    [580, 217], [573, 216], [565, 198], [563, 166], [567, 126],
    [564, 85], [564, 41], [563, 9], [557, 2], [499, 0],
    [431, 0], [369, 2], [335, 0], [295, 1], [263, 3],
    [223, 4], [165, 8], [122, 6], [65, 3], [41, 2],
    [5, 0], [3, 17], [4, 62], [3, 96], [3, 148],
    [3, 186], [5, 224], [5, 250], [8, 278], [4, 306],
    [4, 337], [4, 366], [557, 531],
]


def resize_to_width(image, width):
    height = int(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height))


def create_polygon_mask(shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Smooth mask edges for more natural blending
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask


def apply_semantic_color(base_image, mask, bgr_color, alpha):
    """
    Applies a color overlay to a masked region while preserving
    original grayscale brightness and texture.
    """
    color_layer = np.zeros_like(base_image)
    color_layer[:] = bgr_color

    blended = cv2.addWeighted(base_image, 1 - alpha, color_layer, alpha, 0)

    mask_float = mask.astype(float) / 255.0
    mask_float = mask_float[:, :, None]

    result = base_image * (1 - mask_float) + blended * mask_float
    return result.astype(np.uint8)


def main():
    image = cv2.imread(str(INPUT_IMAGE))

    if image is None:
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    # Resize image to match the coordinate system used when selecting points
    image = resize_to_width(image, DISPLAY_WIDTH)

    # Use grayscale luminance as base
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colorized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Create semantic masks
    sky_mask = create_polygon_mask(colorized.shape, SKY_POINTS)
    sea_mask = create_polygon_mask(colorized.shape, SEA_POINTS)
    boat_mask = create_polygon_mask(colorized.shape, BOAT_POINTS)

    # Apply colors.
    # OpenCV uses BGR format, not RGB.
    colorized = apply_semantic_color(
        colorized,
        sky_mask,
        bgr_color=(210, 190, 160),   # pale blue-gray sky
        alpha=0.30,
    )

    colorized = apply_semantic_color(
        colorized,
        sea_mask,
        bgr_color=(130, 105, 65),    # muted blue-green sea
        alpha=0.35,
    )

    colorized = apply_semantic_color(
        colorized,
        boat_mask,
        bgr_color=(185, 175, 150),   # warm gray / beige boat tone
        alpha=0.18,
    )

    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), colorized)

    print(f"Classical colorized image saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()