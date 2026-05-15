from pathlib import Path
import sys
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEOLDIFY_ROOT = PROJECT_ROOT / "external" / "DeOldify"

INPUT_IMAGE = PROJECT_ROOT / "input" / "image1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_IMAGE = OUTPUT_DIR / "deoldify_colorized.png"

MODEL_SRC = PROJECT_ROOT / "models" / "ColorizeArtistic_gen.pth"
MODEL_DST = DEOLDIFY_ROOT / "models" / "ColorizeArtistic_gen.pth"


def main():
    if not DEOLDIFY_ROOT.exists():
        raise FileNotFoundError(
            "DeOldify repo not found. Run: git clone https://github.com/jantic/DeOldify.git external/DeOldify"
        )

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    if not MODEL_SRC.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_SRC}")

    MODEL_DST.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_DST.exists():
        shutil.copy2(MODEL_SRC, MODEL_DST)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEOLDIFY_ROOT))

    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer
    import torch

    if torch.cuda.is_available():
        device.set(device=DeviceId.GPU0)
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device.set(device=DeviceId.CPU)
        print("Using CPU")

    colorizer = get_image_colorizer(artistic=True)

    result_path = colorizer.plot_transformed_image(
    path=INPUT_IMAGE,
    results_dir=OUTPUT_DIR,
    render_factor=35,
    watermarked=False,
    post_process=True,
)

    result_path = Path(result_path)

    if result_path.exists():
        shutil.copy2(result_path, OUTPUT_IMAGE)
        print(f"Saved colorized image to: {OUTPUT_IMAGE}")
    else:
        print(f"DeOldify finished, but expected result was not found at: {result_path}")


if __name__ == "__main__":
    main()