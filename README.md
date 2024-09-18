# Advanced-Re-Identification-and-Person-Tracking-Using-Multiple-Neural-Network-Architectures

# People Counter Project

The **People Counter** project is an advanced system designed to count and track individuals based on video footage. Utilizing cutting-edge technologies and sophisticated algorithms, the system provides real-time analysis of video streams to accurately detect, track, and re-identify people. Developed in Python, the project integrates several key libraries and models, including YOLO for object detection, SORT for tracking, and SIFT for re-identification.

## Project Overview

The People Counter project addresses the need for robust, real-time people counting and tracking in various applications, including surveillance, crowd management, and analytics. By leveraging state-of-the-art computer vision techniques, the system can efficiently handle high-resolution video inputs, process them in real-time, and deliver accurate results.

### Core Components

1. **Detection**: Using the YOLO (You Only Look Once) model, the system identifies people in video frames. YOLO is known for its speed and accuracy in object detection, allowing the system to process video frames efficiently.

2. **Tracking**: The SORT (Simple Online and Realtime Tracking) algorithm is employed to maintain the identities of detected individuals across multiple frames. This tracking algorithm helps in managing the continuity of identities and reduces errors in tracking.

3. **Re-Identification**: To associate detected individuals with previously observed instances, the system uses feature extraction techniques combined with similarity measures. The SIFT (Scale-Invariant Feature Transform) algorithm and ResNet50 model are used for extracting and comparing features.

## Project Structure

The project is organized into several components and files, each serving a specific function in the overall system:

### 1. Main Script: `People_counter/people_counter.py`

This is the primary script that controls the entire video analysis pipeline. It integrates various stages of processing, including detection, tracking, and re-identification.

- **Description**:
  - **Initialization**: Loads the video file and initializes the YOLO model for object detection.
  - **Processing**: Iterates over each frame in the video, performing the following tasks:
    - **Detection**: Detects people in the frame using YOLO.
    - **Feature Extraction**: Extracts features from detected people using the ResNet50 model.
    - **Tracking**: Tracks detected individuals using the SORT algorithm.
    - **Re-Identification**: Matches new detections with previously tracked individuals based on feature similarity.
    - **Result Saving**: Saves snapshots of detected individuals and the processed video with annotations.

- **Functions**:
  - `extract_reid_features(image)`: Extracts re-identification features from an image using the ResNet50 model. This process involves resizing, normalizing, and preparing the image for feature extraction.
  - `compare_reid_features(features1, features2)`: Calculates the similarity between two feature vectors using cosine similarity. This function helps in matching detected individuals with previously observed ones.
  - `improved_feature_matching(features, persons, similarity_threshold)`: Matches new detections with existing tracked individuals based on feature similarity, refined by a similarity threshold.
  - `compute_cost_matrix(persons, detections)`: Constructs a cost matrix for tracking people, incorporating feature similarity and Intersection over Union (IoU) scores.
  - `main()`: The entry point of the script, orchestrating the entire video processing pipeline, from loading the video to saving results.

### 2. Output Directory: `output_people/`

This directory contains snapshots of detected individuals, organized into folders for each person.

- **Folder Structure**:
  - `person_<id>/`: Each folder corresponds to a detected individual, where `<id>` is a unique identifier assigned to the person.
    - `frame_<frame_number>.jpg`: Images of the person from different video frames. These images are captured and saved for review and analysis.

### 3. Configuration Files

- **`config.yaml`**: Contains configuration settings for the YOLO model, ResNet50, and other parameters. This file allows easy adjustment of model settings and processing options.

## Requirements

To run the People Counter project, ensure you have the following dependencies installed. The project requires Python 3.8 or higher, along with several key libraries and packages. Below is a comprehensive list of required libraries:

### Python Libraries

- **OpenCV**: Provides tools for image and video processing. Install via `opencv-python`.
- **NumPy**: Essential for numerical computations and array manipulations. Install via `numpy`.
- **cvzone**: A library for simple image processing and interactions. Install via `cvzone`.
- **Torch**: The PyTorch library for deep learning and numerical computations. Install via `torch`.
- **Ultralytics (YOLO)**: Implementation of the YOLO object detection model. Install via `ultralytics`.
- **Hydra-Core**: Configuration management library for managing complex configurations. Install via `hydra-core`.
- **Matplotlib**: Used for plotting and visualizing data. Install via `matplotlib`.
- **Pillow**: The Python Imaging Library for image processing tasks. Install via `Pillow`.
- **PyYAML**: YAML parser for handling configuration files. Install via `PyYAML`.
- **Requests**: For making HTTP requests. Install via `requests`.
- **SciPy**: A library for scientific computing, including optimization and integration. Install via `scipy`.
- **scikit-image**: Provides algorithms for image processing. Install via `scikit-image`.
- **filterpy**: Contains tools for filtering and tracking. Install via `filterpy`.
- **lap**: Solves the linear assignment problem. Install via `lap`.

### Sample `requirements.txt`

```plaintext
cvzone==1.5.6
hydra-core>=1.2.0
matplotlib>=3.2.2
numpy>=1.18.5
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.64.0
filterpy==1.4.5
scikit-image==0.19.3
lap==0.4.0
