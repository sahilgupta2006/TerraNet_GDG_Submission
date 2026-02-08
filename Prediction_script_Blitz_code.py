# =====================================================
# TerraSeg – Standalone Inference Script
# =====================================================

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

# =====================================================
# 1. Download model from Google Drive
# =====================================================
import gdown

GDRIVE_URL = "https://drive.google.com/uc?id=1zpFzuF42kDYEvMgm__kAJqgKJtHaNGhp"
MODEL_PATH = "best_model.keras"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
else:
    print("✅ Model already exists")

# =====================================================
# 2. Load model (INFERENCE MODE ONLY)
# =====================================================
model = keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded (compile=False)")

# =====================================================
# 3. CONFIG (MATCHES COLAB NOTEBOOK)
# =====================================================
IMG_SIZE = (512, 512)
NUM_CLASSES = 10

MASK_VALUES = [100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]
CLASS_TO_VALUE = {i: v for i, v in enumerate(MASK_VALUES)}

FG_VALUES = {7100, 10000}
USE_TTA = True

# =====================================================
# 4. Image Loader (same as notebook)
# =====================================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

# =====================================================
# 5. RLE Encoding (unchanged)
# =====================================================
def rle_encode(mask):
    pixels = np.concatenate([[0], mask.flatten(order="F"), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))

# =====================================================
# 6. Prediction with TTA (horizontal flip only)
# =====================================================
def predict_with_tta(model, img):
    img_batch = np.expand_dims(img, axis=0)

    pred1 = model.predict(img_batch, verbose=0)[0]

    if USE_TTA:
        flipped = np.fliplr(img)
        pred2 = model.predict(
            np.expand_dims(flipped, axis=0),
            verbose=0
        )[0]
        pred2 = np.fliplr(pred2)
        pred = (pred1 + pred2) / 2.0
    else:
        pred = pred1

    return np.argmax(pred, axis=-1)

print(f"TTA enabled: {USE_TTA} (horizontal flip only)")

# =====================================================
# 7. Ask user for test image folder
# =====================================================
TEST_IMAGES_DIR = input("📁 Enter full path to TEST images folder: ").strip()

if not os.path.isdir(TEST_IMAGES_DIR):
    raise ValueError("❌ Invalid test image folder path")

test_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
print(f"🧪 Found {len(test_paths)} test images")

# =====================================================
# 8. Run inference
# =====================================================
rows = []

for path in tqdm(test_paths):
    img = load_image(path)
    pred = predict_with_tta(model, img)

    # Resize back to original competition size
    pred = np.array(
        Image.fromarray(pred.astype(np.uint8))
        .resize((960, 540), Image.NEAREST)
    )

    # Convert class → original mask values
    values = np.zeros(pred.shape, dtype=np.uint16)
    for idx, val in CLASS_TO_VALUE.items():
        values[pred == idx] = val

    # Binary foreground mask
    binary = np.isin(values, list(FG_VALUES)).astype(np.uint8)

    rows.append({
        "image_id": os.path.splitext(os.path.basename(path))[0],
        "encoded_pixels": rle_encode(binary)
    })

# =====================================================
# 9. Save submission
# =====================================================
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)

print(f"\n✅ submission.csv saved ({len(df)} rows)")
