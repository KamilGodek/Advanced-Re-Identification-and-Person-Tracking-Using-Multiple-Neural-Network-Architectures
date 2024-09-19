import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import time
from collections import deque

# Initialize TorchReID feature extractor with PCB model
extractor = FeatureExtractor(
    model_name='pcb_p6',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def extract_reid_features(image):
    image = cv2.resize(image, (128, 384))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(extractor.device)
    features = extractor(image)
    return features.cpu().detach().numpy().flatten()

def compare_reid_features(features1, features2):
    return 1 - np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

def improved_feature_matching(features, persons, similarity_threshold):
    matches = []
    for person in persons:
        avg_features = np.mean(person.history, axis=0)
        distance = compare_reid_features(features, avg_features)
        if distance < similarity_threshold:
            matches.append((person, distance))
    matches.sort(key=lambda x: x[1])
    return matches

class Person:
    def __init__(self, id, features, bbox):
        self.id = id
        self.features = features
        self.history = deque([features], maxlen=80)  # Keep last 30 features
        self.last_seen = 0
        self.bbox = bbox
        self.missed_frames = 0

# Initialize YOLO model
model = YOLO('yolov8x.pt')  # Using YOLOv8x for better accuracy
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Set output folder for images
output_folder = "E:/Programowanie/Magisterka/Pic_people"
os.makedirs(output_folder, exist_ok=True)

# Open input video file
cap = cv2.VideoCapture('E:/Programowanie/Magisterka/file_storage/Imagies/video02.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('E:/Programowanie/Magisterka/file_storage/Video/output_video_with_segmentation.mp4', fourcc, fps, (frame_width, frame_height))

persons = []
next_person_id = 1
similarity_threshold = 0.23  # Adjusted for stricter matching
frames_to_keep_person = 8000  # Keep for 1 second assuming 30 fps
MIN_PERSON_AREA = 500  # Reduced to detect smaller persons
MAX_MISSED_FRAMES = 10000  # Maximum number of frames a person can be missed before removing

frame_count = 0
start_time = time.time()
prev_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    results = model(img, classes=[0], conf=0.3)  # Only detect persons with confidence > 0.3

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            if (x2 - x1) * (y2 - y1) > MIN_PERSON_AREA:
                person_img = img[y1:y2, x1:x2]
                try:
                    features = extract_reid_features(person_img)
                    detections.append((x1, y1, x2, y2, features))
                except Exception as e:
                    print(f"Failed to extract features: {e}")
                    continue

    matched_persons = set()

    # Increase missed frames for all persons
    for person in persons:
        person.missed_frames += 1

    for x1, y1, x2, y2, features in detections:
        matches = improved_feature_matching(features, persons, similarity_threshold)

        if matches:
            best_match = matches[0][0]
            best_match.features = 0.7 * best_match.features + 0.3 * features
            best_match.history.append(features)
            best_match.last_seen = frame_count
            best_match.bbox = (x1, y1, x2, y2)
            best_match.missed_frames = 0
            matched_persons.add(best_match.id)
        else:
            new_person = Person(next_person_id, features, (x1, y1, x2, y2))
            next_person_id += 1
            persons.append(new_person)
            matched_persons.add(new_person.id)

    # Remove persons not seen for too long
    persons = [p for p in persons if p.missed_frames < MAX_MISSED_FRAMES]

    # Draw circles (dots) instead of bounding boxes
    for person in persons:
        if person.id in matched_persons:
            x1, y1, x2, y2 = person.bbox
            # Calculate the center of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Draw a dot at the center of the bounding box
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, f'ID: {person.id}', (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display statistics
    cv2.putText(img, f'Current People: {len(matched_persons)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Total Unique People: {next_person_id - 1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(img)

    # Display the image
    cv2.imshow("Person Tracking", img)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# Release everything
cap.release()
out.release()  # Don't forget to release the VideoWriter
cv2.destroyAllWindows()

print(f"Processing time per frame: {(time.time() - start_time) / frame_count:.4f} seconds")
print(f"Total number of unique persons: {next_person_id - 1}")
