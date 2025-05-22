📘 README.md

# 🧠 AI Age Prediction Suite

The **AI Age Prediction Suite** is a powerful web application built using Streamlit that provides real-time age estimation using face detection and deep learning. It allows users to analyze uploaded images, videos, or live webcam input to detect faces and estimate age groups.

## 🚀 Features

- 📷 **Image Mode**: Upload images to detect faces and predict age with confidence scores.
- 🎥 **Video Mode**: Upload videos and process frames in real time for age detection.
- 👁️ **Live Mode**: Use your webcam for live age prediction (requires streamlit-webrtc).
- 📊 **Analytics Tab**: View performance benchmarks and technical insights.

---

## 🧰 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Dhivyalakshmi-M/Age_prediction_using_image_video.git
   cd ai-age-prediction-suite
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download the required models:

Download the following Caffe model files and place them in the root directory:

deploy.prototxt.txt

res10_300x300_ssd_iter_140000.caffemodel

age_deploy.prototxt

age_net.caffemodel

Run the app:

bash
Copy
Edit
streamlit run app.py
📁 Project Structure
bash
Copy
Edit
ai-age-prediction-suite/
│
├── app.py                  # Main Streamlit application
├── style.css              # Custom styles
├── requirements.txt       # Python dependencies
├── deploy.prototxt.txt    # Face detection model prototxt
├── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model weights
├── age_deploy.prototxt    # Age prediction model prototxt
├── age_net.caffemodel     # Age prediction model weights
└── README.md
🧠 Models Used
Face Detection: OpenCV DNN SSD with ResNet base.

Age Prediction: Caffe model trained on IMDB-WIKI dataset.

Age Buckets: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)

📌 Tips for Better Accuracy
Use high-resolution, front-facing images.

Ensure good lighting without facial obstructions.

Avoid sunglasses, hats, or extreme angles.

📄 License
This project is licensed under the MIT License.
