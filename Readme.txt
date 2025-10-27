🧠 Crowd Classifier using Deep Learning
A deep learning–based image classification system that automatically analyzes images to determine crowd density — categorized as Low, Medium, or High.
Built using PyTorch and deployed with Streamlit, this project provides a simple interface for real-time crowd analysis and visualization.

📁 Project Structure
crowd_project/
│
├── .streamlit/              # Streamlit configuration files
├── dataset/                 # Training/testing dataset
├── app.py                   # Main Streamlit app
├── crowd_classifier.pth     # Trained PyTorch model
├── predict.py               # Prediction logic
├── train.py                 # Model training script
├── download_images.py       # Script to collect dataset images
├── requirements.txt         # Python dependencies
├── test.jpeg                # Sample test image 1
├── test2.jpeg               # Sample test image 2
└── __pycache__/             # Cached Python files

⚙️ Installation Steps

Clone the repository
git clone https://github.com/<your-username>/Crowd_Classifier.git
cd Crowd_Classifier


Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux


Install dependencies
pip install -r requirements.txt


Run the Streamlit App
streamlit run app.py

🧩 How It Works
-train.py trains a CNN on crowd images and saves the model as crowd_classifier.pth.
-predict.py loads the trained model and predicts crowd level from new images.
-app.py provides a Streamlit UI where users can upload images and get instant predictions.

🖼️ Output Example
After uploading an image, the app displays:
The input image
Predicted crowd category: Low / Medium / High

🧑‍💻 Technologies Used
Python
PyTorch
Streamlit
OpenCV
NumPy, Pandas

📜 License
This project is open-source under the MIT License.
