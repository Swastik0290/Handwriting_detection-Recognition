# main.py
import argparse
from detection import detect_and_crop
from recognition import recognize_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect then recognize handwritten text.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="ocr_model_50_epoch.h5", help="Path to CRNN .h5 model")
    args = parser.parse_args()

    detect_and_crop(args.image)
    recognize_all(args.model)
