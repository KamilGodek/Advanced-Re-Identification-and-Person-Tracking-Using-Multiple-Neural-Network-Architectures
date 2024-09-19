# Advanced-Re-Identification-and-Person-Tracking-Using-Multiple-Neural-Network-Architectures

# People Counter Project

The **Re-Identification** project is an advanced system designed to count and track individuals based on video footage. Utilizing cutting-edge technologies and sophisticated algorithms, the system provides real-time analysis of video streams to accurately detect, track, and re-identify people. Developed in Python, the project integrates several key libraries and models, including YOLO for object detection, DeepSort for tracking, and SIFT for re-identification.

## Project Overview

The Re-Identification project addresses the need for robust, real-time people counting and tracking in various applications, including surveillance, crowd management, and analytics.
### Core Components

1. **Detection**: Using the YOLO (You Only Look Once) model, the system identifies people in video frames. YOLO is known for its speed and accuracy in object detection, allowing the system to process video frames efficiently.

2. **Tracking**: The Re-Identification project employs the DeepSORT (Deep Simple Online and Realtime Tracking) algorithm to track individuals across video frames. DeepSORT enhances tracking accuracy by combining object detection with appearance-based re-identification, ensuring robust management of individual identities and reducing tracking errors.

3. **Re-Identification**: To associate detected individuals with previously observed instances, the system uses feature extraction techniques combined with similarity measures. The SIFT (Scale-Invariant Feature Transform) algorithm and ResNet50 model are used for extracting and comparing features.

## Project Structure

The project is organized into several components and files, each serving a specific function in the overall system:

### 1. Main Script: `People_counter/OsNet_x1_0.py`

This project is designed for video analysis to detect, track, and re-identify individuals using the YOLO model and the OSNet_x1_0 feature extractor. Below is a detailed description of the project structure.

## Project Structure

The project is divided into several components and files, each serving a specific function within the system:

### 1. Main Script: `people_counter/osnet_x1_0.py`

This is the main script that controls the entire video processing pipeline. It integrates various stages of processing, including detection, tracking, and re-identification.

**Description:**

- **Initialization:**
  - Loads the video file.
  - Initializes the YOLO model for object detection.
  - Initializes the TorchReID feature extractor on GPU.

- **Processing:**
  - Iterates through each frame of the video, performing the following tasks:
    - **Detection:** Detects people in the frame using YOLO.
    - **Feature Extraction:** Extracts features from detected individuals using the OSNet_x1_0 model.
    - **Tracking:** Tracks detected individuals using the assignment algorithm.
    - **Re-identification:** Matches new detections to previously tracked individuals based on feature similarity.
    - **Saving Results:** Saves snapshots of detected individuals and the processed video with annotations.

**Functions:**

- `extract_reid_features(image)`: Extracts re-identification features from an image using the OSNet_x1_0 model. This process includes resizing, normalization, and preparing the image for feature extraction.
- `cosine_similarity(features1, features2)`: Computes the similarity between two feature vectors using cosine similarity. This function helps in matching detected individuals to previously observed ones.
- `calculate_iou(box1, box2)`: Calculates the Intersection over Union (IoU) between two bounding boxes.
- `main()`: The entry point of the script that coordinates the entire video processing pipeline, from loading the video to saving the results.

### 2. Output Directory: `output_people/`

This directory contains snapshots of detected individuals, organized into folders for each person.

**Folder Structure:**

- `person_<id>/`: Each folder corresponds to a detected person, where `<id>` is a unique identifier assigned to that person.
- `frame_<frame_number>.jpg`: Images of the person from different frames of the video. These images are captured and saved for review and analysis.

### 3. Configuration Files

- `config.yaml`: Contains configuration settings for the YOLO model, OSNet_x1_0, and other parameters. This file allows easy adjustment of model settings and processing options.


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

## Main Features and Detailed Functions

### 1. `extract_reid_features(image)`
**Description:** Extracts re-identification features from an image to help match individuals across different frames. This function prepares the image by resizing, converting to RGB, normalizing pixel values, and extracting features using a pre-trained OSNet_x1_0 model.
**Parameters:**
- `image`: The input image from which features are to be extracted.
**Returns:** A feature vector represented as a NumPy array. This vector captures the unique characteristics of the person in the image.

### 2. `cosine_similarity(features1, features2)`
**Description:** Computes the cosine similarity between two feature vectors to determine how similar two detected instances are. This function is crucial for re-identifying people based on their extracted features.
**Parameters:**
- `features1`: Feature vector of the first instance.
- `features2`: Feature vector of the second instance.
**Returns:** A similarity score between 0 and 1, where 1 indicates a perfect match and 0 indicates no similarity.

### 3. `improved_feature_matching(features, persons, similarity_threshold)`
**Description:** Matches newly detected features to currently tracked individuals by comparing feature vectors and evaluating similarity scores against a predefined threshold. This function enhances re-identification accuracy by refining the matching process.
**Parameters:**
- `features`: Feature vector of the newly detected person.
- `persons`: List of currently tracked individuals.
- `similarity_threshold`: Minimum similarity score required for a match.
**Returns:** A list of matched individuals, each associated with a similarity score indicating the degree of match.

### 4. `compute_cost_matrix(persons, detections)`
**Description:** Constructs a cost matrix that combines feature similarity and Intersection over Union (IoU) scores to create a comprehensive metric for matching detections to existing individuals. This matrix is used in the assignment algorithm to determine the best matches.
**Parameters:**
- `persons`: List of currently tracked individuals.
- `detections`: List of newly detected individuals.
**Returns:** A NumPy array representing the cost matrix used to solve the assignment problem.

### 5. `main()`
**Description:** Coordinates the entire video processing pipeline. Performs the following tasks:
- **Loading Video:** Loads the input video file from the specified path.
- **Initializing Models:** Initializes the YOLO model for object detection and the OSNet_x1_0 model for feature extraction.
- **Processing Frames:** Iterates through each video frame, performing detection, feature extraction, and tracking.
- **Tracking and Re-identification:** Utilizes an assignment algorithm based on the Hungarian method (a component of the SORT algorithm) to assign detected objects to existing tracks. The algorithm computes a cost matrix based on feature similarity and Intersection over Union (IoU), and then performs optimal     assignment to track and identify individuals based on the extracted features
- **Saving Results:** Saves annotated video frames and snapshots of detected individuals to the output directory.
- **Real-time Display:** Displays the number of detected individuals in the current frame and the total number of unique individuals detected so far.
**Execution:** Run the script with appropriate parameters to process the video and generate results.


## Additional Information

### Google Drive Results

For a detailed overview of the results produced by the five different code variants, please refer to the following Google Drive link:
https://drive.google.com/drive/folders/189tAR4IfSBas8Gyuk6wVsT4I82gAew8j?usp=drive_link

This link contains:

Annotated Videos: Videos with bounding boxes and identifiers overlaid on detected individuals for each variant.


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
1. **Install Dependencies**: Ensure all required Python libraries are installed. You can use the provided `requirements.txt` file to install dependencies:
To run the Re-Identification project, follow these steps:

1. **Clone the Repository**:
   
     ```bash git clone https://github.com/KamilGodek/Advanced-Re-Identification-and-Person-Tracking-Using-Multiple-Neural-Network-Architectures.git

2. **Navigate to the Project Directory**:

3. **Install Dependencies**:
   
    pip install -r `Requirements.txt`

4. **Configure the Environment**:
   
    Ensure your system has a compatible GPU with CUDA installed for acceleration. If a GPU is not available, the project will default to CPU processing. 
  
5. **Run the Main Script**:

    python people_counter/osnet_x1_0.py



   ```bash


