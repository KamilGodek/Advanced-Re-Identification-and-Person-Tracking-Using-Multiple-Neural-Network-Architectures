# Advanced-Re-Identification-and-Person-Tracking-Using-Multiple-Neural-Network-Architectures

# People Counter Project

The **People Counter** project is an advanced system designed to count and track individuals based on video footage. Utilizing cutting-edge technologies and sophisticated algorithms, the system provides real-time analysis of video streams to accurately detect, track, and re-identify people. Developed in Python, the project integrates several key libraries and models, including YOLO for object detection, SORT for tracking, and SIFT for re-identification.

## Project Overview

The People Counter project addresses the need for robust, real-time people counting and tracking in various applications, including surveillance, crowd management, and analytics. By leveraging state-of-the-art computer vision techniques, the system can efficiently handle high-resolution video inputs, process them in real-time, and deliver accurate results.

### Core Components

1. **Detection**: Using the YOLO (You Only Look Once) model, the system identifies people in video frames. YOLO is known for its speed and accuracy in object detection, allowing the system to process video frames efficiently.

2. **Tracking**: ### **Tracking**: The People Counter project employs the DeepSORT (Deep Simple Online and Realtime Tracking) algorithm to track individuals across video frames. DeepSORT enhances tracking accuracy by combining object detection with appearance-based re-identification, ensuring robust management of individual identities and reducing tracking errors.

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
```

## Key Features and Detailed Functions

### 1. `extract_reid_features(image)`

- **Description**: Extracts distinctive re-identification features from an image to help match individuals across different frames. This function prepares the image by resizing it, converting it to RGB, normalizing pixel values, and extracting features using a pre-trained ResNet50 model.
- **Parameters**:
  - `image`: Input image to be processed for feature extraction.
- **Returns**: A feature vector represented as a NumPy array. This vector captures the unique characteristics of the person in the image.

### 2. `compare_reid_features(features1, features2)`

- **Description**: Computes the cosine similarity between two feature vectors to determine how similar two detected instances are. This function is crucial for re-identifying individuals based on their extracted features.
- **Parameters**:
  - `features1`: Feature vector of the first instance.
  - `features2`: Feature vector of the second instance.
- **Returns**: A similarity score between 0 and 1, where 1 indicates a perfect match and 0 indicates no similarity.

### 3. `improved_feature_matching(features, persons, similarity_threshold)`

- **Description**: Matches newly detected features with existing tracked individuals by comparing feature vectors and evaluating similarity scores against a predefined threshold. This function enhances the accuracy of re-identification by refining the matching process.
- **Parameters**:
  - `features`: Feature vector of the newly detected person.
  - `persons`: List of currently tracked individuals.
  - `similarity_threshold`: Minimum similarity score required for a match.
- **Returns**: A list of matched persons, each associated with a similarity score indicating the degree of match.

### 4. `compute_cost_matrix(persons, detections)`

- **Description**: Constructs a cost matrix that combines feature similarity and IoU (Intersection over Union) scores to create a comprehensive metric for matching detections with existing individuals. This matrix is used in the assignment algorithm to determine the best matches.
- **Parameters**:
  - `persons`: List of currently tracked persons.
  - `detections`: List of newly detected individuals.
- **Returns**: A NumPy array representing the cost matrix used for solving the assignment problem.

### 5. `main()`

- **Description**: The main function that coordinates the entire video processing pipeline. It performs the following tasks:
  - **Video Loading**: Loads the input video file from the specified path.
  - **Model Initialization**: Initializes the YOLO model for object detection and the ResNet50 model for feature extraction.
  - **Frame Processing**: Iterates through each frame of the video, performing detection, feature extraction, and tracking.
  - **Tracking and Re-Identification**: Uses the SORT algorithm to track individuals and re-identifies them based on extracted features.
  - **Result Saving**: Saves annotated video frames and snapshots of detected individuals to the output directory.
  - **Real-Time Display**: Displays the number of people detected in the current frame and the total number of unique individuals detected so far.
- **Execution**: Run the script with the appropriate parameters to process the video and generate results.

## Additional Information

### Automatic Folder Creation

The script automatically creates necessary directories for storing snapshots of detected individuals. This feature simplifies the organization of results and ensures that each detected person has a dedicated folder for storing images from various frames.

### GPU Utilization

If a compatible GPU (CUDA) is available, the script utilizes it to accelerate computations, particularly for model inference and feature extraction. If a GPU is not available, the script defaults to using the CPU, which may result in slower processing times.

### Real-Time Visualization

The script provides real-time feedback by displaying the current count of detected people and the total count of unique individuals detected. This information is overlaid on the video feed, allowing users to monitor the processing and detection results as they occur.

### Result Saving

The processed video, along with snapshots of detected individuals, is saved to the specified output directories. This includes:
- **Annotated Video**: The video with bounding boxes and identifiers overlaid on detected individuals.
- **Snapshots**: Individual images of detected persons, saved in folders named after their unique identifiers.

## Usage Instructions

To run the People Counter project, follow these steps:

1. **Install Dependencies**: Ensure all required Python libraries are installed. You can use the provided `requirements.txt` file to install dependencies:

   ```bash
   pip install -r requirements.txt

