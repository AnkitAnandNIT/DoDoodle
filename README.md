# DoDoodle
My attempt at a simple doodle detector, trained on data from "Quick, Draw!" by Google.


# 🖌️ DoDoodle - Hand Drawn Sketch Classifier

DoDoodle is a web-based application that allows users to draw simple doodles in a browser and get real-time predictions of what they’ve drawn using a Convolutional Neural Network (CNN). It uses a pre-trained model trained on the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset for **Car**, **House**, and **Tree** categories.

![App Screenshot](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Google_Quick_Draw_Dataset.svg/1200px-Google_Quick_Draw_Dataset.svg.png)  
*Image credit: Google QuickDraw dataset*

---

## 🔍 About the Project

This project uses:
- TensorFlow for training a neural network
- Flask for serving a backend API
- HTML5 Canvas and JavaScript for the drawing interface

You draw on a canvas, click **Predict**, and the app tells you what it thinks you drew.

---

## 🚀 Live Features

✅ Draw anything in the browser  
✅ Predicts between **car**, **house**, and **tree**  
✅ Already trained model included  
✅ Option to retrain on other categories if needed  
✅ Clean, responsive interface  

---

## 📦 What's Inside

DoDoodle/
├── app.py # Flask backend
├── train_model.py # Script to train model on QuickDraw .npy files
├── doodle_model.keras # Pre-trained Keras model
├── requirements.txt
├── templates/
│ └── index.html # Frontend HTML (Canvas & JS)
├── static/
│ └── script.js # JS if separated (you can add script if you want, I've not attached any)
├── README.md



---

## 🧠 Model Details

- The current `doodle_model.keras` file is trained on:
  - `full_numpy_bitmap_car.npy`
  - `full_numpy_bitmap_house.npy`
  - `full_numpy_bitmap_tree.npy`

These are part of Google’s **QuickDraw Dataset**, which contains millions of 28x28 hand-drawn doodles across hundreds of categories.

### 📚 Dataset Source

> **Quick, Draw! Dataset**  
> Provided by Google Creative Lab  
> [Download the dataset here →](https://github.com/googlecreativelab/quickdraw-dataset)

---

## 💻 How to Use

### Step 1: Clone the Repo

```bash:
git clone https://github.com/YOUR_USERNAME/DoDoodle.git
cd DoDoodle
Step 2: Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Step 3: Run the App
bash
Copy
Edit
python app.py

Then open your browser at:
👉 http://127.0.0.1:5000
```
🧪 Training the Model (Optional)
If you'd like to train the model on different classes:

1. Download the .npy files
Visit: https://github.com/googlecreativelab/quickdraw-dataset

Example:
full_numpy_bitmap_cat.npy
full_numpy_bitmap_dog.npy

2. Replace or add new files in your project folder.
3. Update the train_model.py script:

data_files = [
    ("full_numpy_bitmap_cat.npy", "cat"),
    ("full_numpy_bitmap_dog.npy", "dog"),
]
Then train with:

bash:
python train_model.py
This will create a new doodle_model.keras file.

🤝 Acknowledgments
✏️ Dataset by Google Creative Lab — Quick, Draw!

💻 Built with TensorFlow, Flask, HTML5 & JavaScript

📃 License
This project is under the MIT License — free to use and modify, just give credit where it's due!

🔗 Connect With Me
If you liked this, feel free to connect on LinkedIn or ⭐ the repo!


