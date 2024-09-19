import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import time
from scipy.optimize import linear_sum_assignment

# Configuration
OUTPUT_FOLDER = "output_people"
YOLO_MODEL_PATH = 'E:/Programowanie/Magisterka/file_storage/YOLO-Size/yolov8x.pt'

# Adjusted parameters
SIMILARITY_THRESHOLD = 0.6
MAX_HISTORY_LENGTH = 50
FRAMES_TO_KEEP_PERSON = 30
CONFIDENCE_THRESHOLD = 0.5
FEATURE_UPDATE_WEIGHT = 0.3
IOU_THRESHOLD = 0.3
MAX_COST = 1000000

# Initialize TorchReID feature extractor
extractor = FeatureExtractor(
    model_name='resnet50',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


def extract_reid_features(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(extractor.device)
    features = extractor(image)
    return features.cpu().detach().numpy().flatten()


def cosine_similarity(features1, features2):
    return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


class Person:
    def __init__(self, id, features, bbox):
        self.id = id
        self.features = features
        self.bbox = bbox
        self.last_seen = 0
        self.track_history = []

    def update(self, features, bbox, frame_count):
        self.features = features
        self.bbox = bbox
        self.last_seen = frame_count
        self.track_history.append(bbox)
        if len(self.track_history) > MAX_HISTORY_LENGTH:
            self.track_history.pop(0)


def compute_cost_matrix(persons, detections):
    cost_matrix = np.zeros((len(persons), len(detections)))
    for i, person in enumerate(persons):
        for j, (_, _, _, _, features, _, _) in enumerate(detections):
            feature_similarity = cosine_similarity(person.features, features)
            iou = calculate_iou(person.bbox, detections[j][:4])
            cost = 1 - (feature_similarity * 0.7 + iou * 0.3)  # Weighted combination
            cost_matrix[i, j] = cost if cost < SIMILARITY_THRESHOLD else MAX_COST
    return cost_matrix


def main():
    model = YOLO(YOLO_MODEL_PATH)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture('E:/Programowanie/Magisterka/file_storage/Imagies/video02.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter('E:/Programowanie/Magisterka/file_storage/Video/output_video.mp4', fourcc, fps,
                          (frame_width, frame_height))

    persons = []
    next_person_id = 1
    frame_count = 0
    prev_time = time.time()

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("End of video stream or error reading frame.")
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = model(img)

            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    if cls == 0 and conf > CONFIDENCE_THRESHOLD:
                        person_img = img[y1:y2, x1:x2]
                        features = extract_reid_features(person_img)
                        detections.append((x1, y1, x2, y2, features, person_img, conf))

            if persons and detections:
                cost_matrix = compute_cost_matrix(persons, detections)
                person_indices, detection_indices = linear_sum_assignment(cost_matrix)

                matched_indices = [(i, j) for i, j in zip(person_indices, detection_indices) if
                                   cost_matrix[i, j] < SIMILARITY_THRESHOLD]
                unmatched_persons = [i for i in range(len(persons)) if i not in person_indices]
                unmatched_detections = [j for j in range(len(detections)) if j not in detection_indices]
            else:
                matched_indices = []
                unmatched_persons = list(range(len(persons)))
                unmatched_detections = list(range(len(detections)))

            # Update matched persons
            for person_idx, detection_idx in matched_indices:
                x1, y1, x2, y2, features, _, _ = detections[detection_idx]
                persons[person_idx].update(features, (x1, y1, x2, y2), frame_count)

            # Add new persons for unmatched detections
            for detection_idx in unmatched_detections:
                x1, y1, x2, y2, features, _, _ = detections[detection_idx]
                new_person = Person(next_person_id, features, (x1, y1, x2, y2))
                next_person_id += 1
                new_person.last_seen = frame_count
                persons.append(new_person)

            # Remove persons not seen for a while
            persons = [p for p in persons if frame_count - p.last_seen <= FRAMES_TO_KEEP_PERSON]

            # Visualize and save results
            for person in persons:
                if frame_count - person.last_seen <= 1:  # Only draw recently seen persons
                    x1, y1, x2, y2 = person.bbox

                    # Obliczanie środka prostokąta (cx, cy)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Rysowanie kropki na środku osoby
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    # Rysowanie ID osoby obok kropki
                    cv2.putText(img, f'ID: {person.id}', (cx, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Zapisz wizerunki osób
                    person_folder = os.path.join(OUTPUT_FOLDER, f"person_{person.id}")
                    os.makedirs(person_folder, exist_ok=True)
                    person_img = img[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(person_folder, f"frame_{frame_count}.jpg"), person_img)

            cv2.putText(img, f'Current People: {len(persons)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Unique People: {next_person_id - 1}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(img, f'FPS: {int(fps)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            out.write(img)

            cv2.imshow("Image ReID", img)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        end_time = time.time()
        if frame_count > 0:
            print(f"Average processing time per frame: {(end_time - prev_time) / frame_count:.4f} seconds")
        else:
            print("No frames were processed.")


if __name__ == "__main__":
    main()
