# Tennis Analysis

## Overview
Welcome to Tennis Analysis project! 🎾 This is a project I've been working on to dive deeper into the world of sports analytics, specifically focusing on tennis. Leveraging advanced machine learning, deep learning and computer vision techniques, I've developed a system to analyze tennis players' performance in videos, providing insights into their speed, ball shot speed, and shot count. This project detects players and tennis ball using YOLO (You Only Look Once) and extracts court keypoints with Convolution Neural Networks (CNNs).

## Demo
Check out this snapshot from an output video:
<img width="1941" alt="image" src="https://github.com/toan-ly/Tennis-Analysis/assets/104543062/5588b5c2-cf72-4eba-ab51-d3e025414dd4">

## Models Used
Here are the models I've used in this project:
#### Player Detection
* YOLO v8 for player detection
#### Tennis Ball Detection
* Fine-tuned YOLO v5 for tennis ball detection
#### Court Keypoint Extraction
* ResNet50 for tennis court keypoint

## Training
If you're interested in how I trained these models, feel free to check out the notebooks:
* Tennis ball detector: training/tennis_ball_detector.ipynb
* Tennis court keypoints detector: training/tennis_court_keypoints_training.ipynb

Models are saved here:
* Tennis ball detector model: https://drive.google.com/file/d/1Mn7YgqC76DxXuV6xg22gj-779zl4gVFp/view?usp=sharing
* Tennis court keypoints detector model: https://drive.google.com/file/d/1rh1oKWsFj_sJhY3-gUUKkgyYt3U2PMY_/view?usp=sharing

## Requirements
Here's what you'll need to run this project:
* Python 3.8
* Ultralytics
* PyTorch
* Pandas
* NumPy
* OpenCV

## License
This project is licensed under the MIT License.
