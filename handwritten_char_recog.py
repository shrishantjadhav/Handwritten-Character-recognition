import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import string
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "handwritten_alphabet_model.h5"
letters = list(string.ascii_uppercase)  # EMNIST uses uppercase A–Z

# Create debug folder if not present
os.makedirs("debug_frames", exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully:", MODEL_PATH)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit, 's' to save debug image.")

# ---------------- PREPROCESS HELPERS ----------------
def deskew_and_center(img28):
    # Deskew image based on its central moments
    m = cv2.moments(img28)
    if abs(m['mu02']) < 1e-2:
        return img28
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 28 * skew],
                    [0, 1, 0]])
    img = cv2.warpAffine(img28, M, (28, 28),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return img
    cx, cy = np.mean(xs), np.mean(ys)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    M2 = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img = cv2.warpAffine(img, M2, (28, 28),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return img


def preprocess_roi(roi_gray):
    """Preprocess ROI (grayscale) to 28x28 EMNIST style image."""
    roi = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 8)

    # Morphological cleanup
    roi = cv2.erode(roi, None, iterations=1)
    roi = cv2.dilate(roi, None, iterations=1)

    contours, _ = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(roi, (28, 28))

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(roi.shape[1], x + w + padding)
    y2 = min(roi.shape[0], y + h + padding)
    crop = roi[y1:y2, x1:x2]

    # Resize keeping aspect ratio
    h2, w2 = crop.shape
    if h2 > w2:
        new_h, new_w = 20, int(round(w2 * (20.0 / h2)))
    else:
        new_w, new_h = 20, int(round(h2 * (20.0 / w2)))
    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Deskew + center
    canvas = deskew_and_center(canvas)
    return canvas


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Define a clear ROI (draw a box on screen)
    x1, y1, x2, y2 = 200, 100, 450, 350
    roi_gray = gray[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess
    proc28 = preprocess_roi(roi_gray)

    # EMNIST orientation correction (A-Z uppercase)
    proc28_oriented = cv2.transpose(proc28)
    proc28_oriented = cv2.flip(proc28_oriented, 1)

    # Make sure letter = white, background = black
    if np.mean(proc28_oriented) < 127:
        proc28_oriented = cv2.bitwise_not(proc28_oriented)

    # Normalize for model
    X = proc28_oriented.astype('float32') / 255.0
    X = np.expand_dims(X, axis=(0, -1))  # (1,28,28,1)

    # Predict
    preds = model.predict(X, verbose=0)[0]
    predicted_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100.0
    predicted_letter = letters[predicted_index] if predicted_index < len(letters) else "?"

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, f"{predicted_letter} ({confidence:.1f}%)",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Processed 28x28", cv2.resize(proc28_oriented, (280, 280), interpolation=cv2.INTER_NEAREST))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"debug_frames/{predicted_letter}_{int(confidence)}.png"
        cv2.imwrite(fname, proc28_oriented)
        print("💾 Saved:", fname)

cap.release()
cv2.destroyAllWindows()
