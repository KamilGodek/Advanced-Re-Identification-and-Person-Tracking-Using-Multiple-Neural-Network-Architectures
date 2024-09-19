import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from omegaconf import OmegaConf
import argparse

# Add the path to the folder containing LUPerson-NL
sys.path.append('E:/Programowanie/Magisterka/People_counter/DZIALAJACE_PROBY/LUPerson-NL')

from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights


def find_config_file(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


# Initialize LUPerson-NL model
def init_luperson_model(config_path, checkpoint_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    model = build_model(cfg.model)
    load_pretrained_weights(model, checkpoint_path)
    model = model.cuda()
    model.eval()
    return model


# Extract ReID features
def extract_reid_features(model, image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()


# Compare ReID features
def compare_reid_features(features1, features2):
    return 1 - np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))


class Person:
    def __init__(self, id, features):
        self.id = id
        self.features = features
        self.history = [features]
        self.last_seen = 0


def main(config_path, checkpoint_path, yolo_model_path, video_path, output_folder):
    # Find config file if not found at the specified path
    if not os.path.exists(config_path):
        base_path = 'E:/Programowanie/Magisterka/People_counter/DZIALAJACE_PROBY/LUPerson-NL/fast-reid/configs'
        config_file = find_config_file(base_path, 'bagtricks_R50-ibn.yml')
        if config_file:
            config_path = config_file
            print(f"Config file found at: {config_path}")
        else:
            print(f"Error: Config file 'bagtricks_R50-ibn.yml' not found in {base_path} or its subdirectories")
            return

    # Initialize models
    try:
        luperson_model = init_luperson_model(config_path, checkpoint_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model not found at {yolo_model_path}")
        return

    yolo_model = YOLO(yolo_model_path)
    yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Open video capture
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    persons = []
    next_person_id = 0
    similarity_threshold = 0.1
    max_history_length = 300
    frames_to_keep_person = 5000
    frame_count = 0
    start_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        results = yolo_model(img)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0 and conf > 0.4:
                    person_img = img[y1:y2, x1:x2]
                    features = extract_reid_features(luperson_model, person_img)
                    detections.append((x1, y1, x2, y2, features, person_img))

        matched_persons = set()

        for x1, y1, x2, y2, features, person_img in detections:
            best_match = None
            min_distance = float('inf')

            for person in persons:
                distance = compare_reid_features(features, person.features)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person

            if best_match is None or min_distance > similarity_threshold:
                new_person = Person(next_person_id, features)
                next_person_id += 1
                persons.append(new_person)
                best_match = new_person
                print(f"New person detected. Assigned ID: {new_person.id}")
            else:
                best_match.features = 0.9 * best_match.features + 0.1 * features
                best_match.history.append(features)
                if len(best_match.history) > max_history_length:
                    best_match.history.pop(0)

            best_match.last_seen = frame_count
            matched_persons.add(best_match.id)

            person_folder = os.path.join(output_folder, f"person_{best_match.id}")
            os.makedirs(person_folder, exist_ok=True)
            cv2.imwrite(os.path.join(person_folder, f"frame_{frame_count}.jpg"), person_img)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {best_match.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        persons = [p for p in persons if p.id in matched_persons or frame_count - p.last_seen < frames_to_keep_person]

        cv2.imshow("Image ReID", img)

        print(f"Frame {frame_count}: Active persons: {len(persons)}, Total unique persons: {next_person_id}")

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Processing time per frame: {(end_time - start_time) / frame_count:.4f} seconds")
    print(f"Total number of unique persons: {next_person_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Re-identification with YOLO and LUPerson-NL")
    parser.add_argument("--config",
                        default="E:/Programowanie/Magisterka/People_counter/DZIALAJACE_PROBY/LUPerson-NL/fast-reid/configs/Market1501/bagtricks_R50-ibn.yml",
                        help="Path to LUPerson-NL config file")
    parser.add_argument("--checkpoint",
                        default="E:/Programowanie/Magisterka/People_counter/DZIALAJACE_PROBY/LUPerson-NL//model//model.pth",
                        help="Path to LUPerson-NL checkpoint file")
    parser.add_argument("--yolo", default="E:/Programowanie/Magisterka/file_storage/YOLO-Size/yolov8x.pt",
                        help="Path to YOLO model file")
    parser.add_argument("--video", default="E:/Programowanie/Magisterka/file_storage/Imagies/MagisterkaCPY1.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", default="E:/Programowanie/Magisterka/Pic_people", help="Path to output folder")
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.yolo, args.video, args.output)