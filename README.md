# Kratos 💪: Your AI Fitness Trainer

Kratos is a cutting-edge, real-time pose correction application designed to be your personal AI fitness coach. Using just your webcam, Kratos analyzes your form during exercises and provides immediate, actionable feedback to help you perform movements correctly, prevent injuries, and maximize your workout results.

<br>

<img width="1920" height="1080" alt="Screenshot_2025-11-09_19_37_41" src="https://github.com/user-attachments/assets/40d9240a-46ca-4141-b826-cce8f70dd1aa" />


**[Watch the "AI Fitness Trainer | Demonstration" on YouTube](https://www.youtube.com/watch?v=AsryZZR7MzA)**

<br>

## 🌟 Key Features

* **Real-Time Pose Correction:** Get instant feedback on your form as you exercise.
* **AI-Powered Analysis:** Utilizes machine learning and computer vision to understand your body's posture.
* **Multiple Exercise Modules:** Guided training for **Squats**, **Lunges**, **Bicep Curls**, and **Planks**.
* **Repetition Counting:** Automatically counts your reps for exercises like lunges and squats.
* **Detailed Error Detection:** Identifies common mistakes, such as knee-over-toe in lunges or improper squat depth.
* **Embedded Video Tutorials:** Watch professional tutorials for each exercise before you start.
* **Browser-Based:** Runs entirely in your web browser using TensorFlow.js—no installation required.

## 🛠️ Tech Stack

* **Frontend:** TensorFlow.js (`tf-js`), React
* **ML Model Training:** Python, TensorFlow, Keras, Scikit-learn
* **Pose Estimation:** MediaPipe
* **Data Science:** Pandas, NumPy, Scikit-learn (StandardScaler, LogisticRegression)
* **Model Storage:** Pickle

## ⚙️ How It Works

Kratos uses a sophisticated pipeline to analyze your form in real-time.

1.  **Pose Estimation:** The webcam feed is processed by **MediaPipe** to extract 3D pose landmarks for your entire body.
2.  **Data Preprocessing:** These landmarks (x, y, z, visibility) are flattened and scaled in real-time.
3.  **Real-time Inference:** A custom-trained model, converted to **TensorFlow.js** format, runs in the browser. It predicts your current pose or exercise stage from the landmark data.
4.  **Logic & Feedback:**
    * For exercises like **Squats**, a **Logistic Regression** model classifies your state (e.g., 'up' or 'down').
    * For complex exercises like **Lunges**, a hybrid system is used:
        * A machine learning model predicts the **stage** (e.g., 'init', 'mid', 'down').
        * A second model predicts **errors** (e.g., 'correct' vs. 'knee-over-toe').
        * Heuristic angle calculations (using `calculate_angle`) provide further validation on knee angles.
    * A state machine in the frontend manages rep counting and provides the correct feedback based on the model's output.

<img width="1920" height="1080" alt="Screenshot_2025-11-09_19_37_41" src="https://github.com/user-attachments/assets/19725c92-6665-4f10-9591-33de750ca1df" />

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Node.js & npm (or yarn)
* A modern web browser with camera access

### Installation

1.  Clone the repo:
    ```sh
    git clone [https://github.com/sunnyallana/kratos.git](https://github.com/sunnyallana/kratos.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd kratos
    ```
3.  Install NPM packages:
    ```sh
    npm install
    ```
4.  Run the development server (this project was seen running on port 5175, common for Vite):
    ```sh
    npm run dev
    ```
5.  Open [http://localhost:5175](http://localhost:5175) in your browser.

## 🏋️ How to Use

1.  Open the application in your browser.
2.  Select an exercise from the main menu.
3.  Allow the app to use your webcam when prompted.
4.  Watch the video tutorial to understand the correct form.
5.  Position yourself so your full body is visible, following the "AI Feedback" panel.
6.  Press **Start** and begin your exercise.
7.  Follow the real-time feedback to adjust your form and maximize your workout!

## 🤖 Model Training

The AI models used in this project were trained in Python. The complete training scripts (like `SquatPostureMLPipeline` and `LungeDetection`) can be found in the `/model_training` directory (you can create this directory).

Our approach involved:
* **Data Collection:** Gathering and labeling video data for each exercise.
* **Landmark Extraction:** Using MediaPipe to process videos and extract landmark coordinates into CSV files.
* **Model Prototyping:**
    * **Scikit-learn:** We used `LogisticRegression` for simple classification tasks (e.g., Squat 'up'/'down') which proved highly effective and lightweight.
    * **Deep Learning:** We also experimented with 3-7 layer deep neural networks (using Keras/TensorFlow) with and without dropout for more complex error detection.
* **Model Export:** The final, trained models (e.g., `.pkl` files) and their `StandardScaler` objects were saved.
* **Conversion:** These models were then converted to TensorFlow.js format (`model.json` and weight files) for use in the frontend.

## 📜 License

Distributed under the MIT License. See `LICENSE.md` for more information.

## 🤝 Acknowledgements

* [MediaPipe](https://mediapipe.dev/)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
