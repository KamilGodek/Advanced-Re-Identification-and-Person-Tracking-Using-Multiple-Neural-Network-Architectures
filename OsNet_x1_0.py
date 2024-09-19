import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import time
from scipy.optimize import linear_sum_assignment

# Inicjalizacja ekstraktora cech TorchReID na GPU
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# Funkcja ekstrakcji cech reidentyfikacyjnych osoby
def extract_reid_features(image):
    image = cv2.resize(image, (256, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(extractor.device)
    features = extractor(image)
    return features.cpu().detach().numpy().flatten()


# Funkcja licząca podobieństwo kosinusowe
def cosine_similarity(features1, features2):
    return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))


# Inicjalizacja modelu YOLO na GPU
model = YOLO('E:/Programowanie/Magisterka/file_storage/YOLO-Size/yolov8x.pt')
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Folder wyjściowy
output_folder = "E:/Programowanie/Magisterka/Pic_people"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture('E:/Programowanie/Magisterka/file_storage/Imagies/video02.mp4')

# Inicjalizacja VideoWriter do zapisu przetworzonego wideo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
out = cv2.VideoWriter('E:/Programowanie/Magisterka/file_storage/Video/Osnet_x1_0_improved.mp4', fourcc, fps,
                      (frame_width, frame_height))


class Person:
    def __init__(self, id, features, bbox):
        self.id = id
        self.features = features
        self.history = [features]
        self.last_seen = 0
        self.confidence = 1.0
        self.bbox = bbox
        self.track = [bbox]
        self.avg_features = features


persons = []
next_person_id = 0

# Zmodyfikowane parametry algorytmu
SIMILARITY_THRESHOLD = 0.25  # Zwiększono próg podobieństwa
FEATURE_UPDATE_WEIGHT = 0.05  # Zmniejszono wagę aktualizacji cech
CONFIDENCE_DECAY = 0.99  # Zwiększono spadek pewności
MIN_CONFIDENCE_TO_KEEP = 0.5  # Zwiększono minimalną pewność do zachowania osoby
FRAMES_TO_KEEP_PERSON = 150  # Zmniejszono liczbę klatek do zachowania osoby
MAX_HISTORY_LENGTH = 50  # Zmniejszono maksymalną długość historii
IOU_THRESHOLD = 0.5  # Zwiększono próg IoU
MAX_COST = 0.7  # Maksymalny koszt do przypisania osoby

frame_count = 0
start_time = time.time()
prev_time = start_time

unique_person_ids = set()


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    results = model(img)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            if cls == 0 and conf > 0.4:  # Wykrywanie ludzi (klasa 0)
                person_img = img[y1:y2, x1:x2]
                features = extract_reid_features(person_img)
                detections.append((x1, y1, x2, y2, features, person_img, conf))

    # Tworzenie macierzy kosztów
    cost_matrix = np.zeros((len(detections), len(persons)))
    for i, detection in enumerate(detections):
        for j, person in enumerate(persons):
            x1, y1, x2, y2, features, _, _ = detection
            bbox = (x1, y1, x2, y2)
            iou = calculate_iou(bbox, person.bbox)
            similarity = cosine_similarity(features, person.avg_features)
            cost = 1 - (0.3 * iou + 0.7 * similarity)  # Zwiększono wagę podobieństwa cech
            cost_matrix[i, j] = cost

    # Wykonanie przypisania Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_persons = set()
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < MAX_COST:
            x1, y1, x2, y2, features, person_img, conf = detections[i]
            person = persons[j]

            # Aktualizacja cech istniejącej osoby
            weight = conf * FEATURE_UPDATE_WEIGHT
            person.features = (1 - weight) * person.features + weight * features
            person.history.append(features)
            if len(person.history) > MAX_HISTORY_LENGTH:
                person.history.pop(0)
            person.avg_features = np.mean(person.history, axis=0)
            person.confidence = max(person.confidence * CONFIDENCE_DECAY, conf)
            person.last_seen = frame_count
            person.bbox = (x1, y1, x2, y2)
            person.track.append((x1, y1, x2, y2))

            matched_persons.add(person.id)
        else:
            # Jeśli koszt jest zbyt wysoki, traktujemy to jako nową osobę
            x1, y1, x2, y2, features, person_img, conf = detections[i]
            new_person = Person(next_person_id, features, (x1, y1, x2, y2))
            next_person_id += 1
            persons.append(new_person)
            matched_persons.add(new_person.id)
            unique_person_ids.add(new_person.id)
            print(f"New person detected. Assigned ID: {new_person.id}")

    # Dodawanie nowych osób, które nie zostały dopasowane
    for i in range(len(detections)):
        if i not in row_ind:
            x1, y1, x2, y2, features, person_img, conf = detections[i]
            new_person = Person(next_person_id, features, (x1, y1, x2, y2))
            next_person_id += 1
            persons.append(new_person)
            matched_persons.add(new_person.id)
            unique_person_ids.add(new_person.id)
            print(f"New person detected. Assigned ID: {new_person.id}")

    # Usuwanie osób, które zniknęły z kadru
    persons = [p for p in persons if p.id in matched_persons or (
                frame_count - p.last_seen < FRAMES_TO_KEEP_PERSON and p.confidence > MIN_CONFIDENCE_TO_KEEP)]

    # Rysowanie wyników na obrazie
    for person in persons:
        if person.id in matched_persons:
            x1, y1, x2, y2 = person.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Rysowanie kropki na środku osoby
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Rysowanie ID osoby
            cv2.putText(img, f"ID: {person.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame_person_count = len(matched_persons)

    # Obliczanie i wyświetlanie FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Wyświetlanie liczby osób i FPS
    cv2.putText(img, f'Current People: {frame_person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Unique People: {len(unique_person_ids)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f'FPS: {int(fps)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Wyświetlanie obrazu
    cv2.imshow("Image ReID", img)

    # Zapis klatki do pliku wideo
    out.write(img)

    print(f"Frame {frame_count}: Active persons: {frame_person_count}, Total unique persons: {len(unique_person_ids)}")

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
out.release()  # Zamknięcie pliku wideo
cv2.destroyAllWindows()

end_time = time.time()
print(f"Czas przetwarzania na klatkę: {(end_time - start_time) / frame_count:.4f} sekundy")
print(f"Całkowita liczba unikalnych osób: {len(unique_person_ids)}")