# ✍️ Handwritten Letter Recognition using CNN

## 📘 Project Overview
This project focuses on building a **Handwritten Letter Recognition System** using **Convolutional Neural Networks (CNN)** trained on the **EMNIST Letters Dataset**.  
The system can recognize **A–Z alphabets** from handwritten images and classify them into their respective letters.  
It also supports **real-time detection** via webcam using **OpenCV**.

---

## 🎯 Objectives
- Develop a deep learning model that recognizes handwritten letters (A–Z).  
- Use CNN architecture to achieve high accuracy and reliability.  
- Enable real-time recognition using a webcam.  
- Support applications in education, digitization, and assistive tools.

---

## 🧠 Technologies Used
- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - TensorFlow / Keras  
  - NumPy  
  - OpenCV  
  - Matplotlib  
  - String, OS  
- **Dataset:** [EMNIST Letters Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing
- Load and normalize the EMNIST dataset.  
- Convert images to 28x28 grayscale arrays.  
- Split data into training and testing sets.

### 2️⃣ Model Building (CNN)
- Build a CNN model with:
  - Conv2D and MaxPooling2D layers
  - Flatten + Dense + Dropout layers  
- Compile using:
  ```python
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
### 3️⃣ Model Training

- Train for 10–15 epochs with batch normalization.

- Evaluate accuracy on the test dataset.

### 4️⃣ Real-time Testing

## Capture webcam feed using OpenCV.

- Detect and preprocess handwritten input (ROI extraction, thresholding, resizing).

- Predict and display the recognized letter in real time.

### 🧩 Features

✅ Recognizes handwritten alphabets (A–Z)
✅ High accuracy using CNN-based deep learning
✅ Real-time recognition using webcam
✅ Simple and modular Python implementation
✅ Easy to extend for digits or full words

### 📊 Model Performance
## Metric	Value
- Training Accuracy	~97%
- Testing Accuracy	~95%
## Loss	Very Low

### 💻 How to Run
```
Step 1: Install Dependencies
pip install tensorflow opencv-python numpy matplotlib

Step 2: Train the Model (Optional)

If you want to retrain:

python handwritten_alphabet_train.py

Step 3: Run Real-time Recognition
python handwritten_alphabet_live.py
```

```
📁 Project Structure
├── handwritten_alphabet_train.py      # CNN training script
├── handwritten_alphabet_live.py       # Live recognition script
├── handwritten_alphabet_model.h5      # Trained model file
├── dataset/                           # EMNIST dataset (if stored locally)
├── debug_images/                      # Optional saved test images
└── README.md                          # Project documentation
```

### 🧩 Future Scope

Extend model to digits and symbols.

Recognize complete words and sentences using NLP.

Develop a mobile or web app version.

Integrate text-to-speech for accessibility.
