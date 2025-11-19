import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ==== PATHS ====
CROPS_DIR = "output/crops"
OUTPUT_FILE = "output/recognized_text.txt"

# ==== LOAD MODEL ====
print("Loading CRNN (TrOCR) model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# ==== ENSURE OUTPUT DIR EXISTS ====
os.makedirs("output", exist_ok=True)

# ==== PROCESS EACH CROPPED IMAGE ====
results = []

print("\nRecognizing text from crops...")
for filename in sorted(os.listdir(CROPS_DIR)):
    img_path = os.path.join(CROPS_DIR, filename)
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append((filename, text))
        print(f"{filename} -> {text}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ==== SAVE RESULTS ====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for filename, text in results:
        f.write(f"{filename}: {text}\n")

print("\n✅ Recognition complete!")
print(f"📄 Output saved to: {OUTPUT_FILE}")
