import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os

# Sprawdzanie dostępności GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funkcja do segmentacji z użyciem DeepLabV3
def segment_person(image, model, device):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    output_predictions = output.argmax(0).cpu().numpy()

    # Zwracamy binarną maskę, gdzie wartość 15 odpowiada osobom
    mask = (output_predictions == 15).astype(np.uint8)
    return mask

# Inicjalizacja modeli na GPU
segmentation_model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1').to(device).eval()
model = YOLO('E:/Programowanie/Magisterka/file_storage/YOLO-Size/yolov8x.pt').to(device)

# Inicjalizacja DeepSORT z obsługą GPU
tracker = DeepSort(
    max_age=12000,
    nn_budget=120,
    max_cosine_distance=0.2,
    embedder_gpu=True,
    half=False,
    embedder="mobilenet"
)

# Wczytanie maski
mask = cv2.imread('E:/Programowanie/Magisterka/file_storage/Imagies/TAK.png', cv2.IMREAD_GRAYSCALE)

# Wczytanie wideo i przygotowanie do zapisu wynikowego wideo
cap = cv2.VideoCapture('E:/Programowanie/Magisterka/file_storage/Imagies/video02.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
mask = cv2.resize(mask, (frame_width, frame_height))

# Ustawienie ścieżki zapisu wynikowego wideo
out = cv2.VideoWriter('E:/Programowanie/Magisterka/file_storage/Video/output_video_with_segmentation.mp4', fourcc, fps, (frame_width, frame_height))

prev_time = 0

# Zmienna do przechowywania unikalnych ID osób
unique_person_ids = set()

# Katalog do zapisywania zdjęć osób
output_folder = 'E:/Programowanie/Magisterka/Pic_people'
os.makedirs(output_folder, exist_ok=True)

# Główna pętla przetwarzania wideo
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_clean = img.copy()

    # Nałożenie maski na obraz, żeby ograniczyć obszar detekcji i segmentacji
    masked_img = cv2.bitwise_and(img_clean, img_clean, mask=mask)

    # Wykrywanie osób za pomocą YOLOv8 (na GPU)
    result = model(masked_img, stream=True)
    detections = []  # Lista detekcji dla DeepSORT
    frame_person_count = 0  # Licznik osób na bieżącej klatce

    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Współrzędne dwóch rogów prostokąta
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.4:  # Jeśli obiekt to osoba
                person_image = img_clean[y1:y2, x1:x2]

                # Segmentacja osoby z użyciem DeepLabV3 (na GPU)
                person_mask = segment_person(person_image, segmentation_model, device)

                # Stosowanie maski do wyodrębnienia osoby
                person_segmented = cv2.bitwise_and(person_image, person_image, mask=person_mask)

                # Opcjonalnie, pokazujemy maskowaną osobę na obrazie
                img_clean[y1:y2, x1:x2] = person_segmented

                # Przekształcenie współrzędnych do formatu [x1, y1, w, h]
                w, h = x2 - x1, y2 - y1
                bbox = [x1, y1, w, h]

                # Dodajemy detekcję do listy dla DeepSORT w odpowiednim formacie
                detections.append((bbox, conf, "person"))  # Poprawny format dla DeepSORT

    # Debugowanie detekcji
    if len(detections) > 0:
        print("Detections:", detections)

    # Sprawdzamy, czy są jakiekolwiek detekcje przed przekazaniem ich do trackera
    if len(detections) > 0:
        # Aktualizacja śledzenia z DeepSORT
        tracks = tracker.update_tracks(detections, frame=img_clean)

        # Zliczanie osób na bieżącej klatce oraz zbieranie unikalnych ID
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Zliczanie osób na bieżącej klatce
            frame_person_count += 1

            # Dodanie unikalnego ID osoby do zbioru
            unique_person_ids.add(track.track_id)

            # Pobieranie współrzędnych obramowania
            x1, y1, x2, y2 = track.to_tlbr().astype(int)

            # Obliczanie środka prostokąta (cx, cy)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Rysowanie kropki na środku osoby
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Wyświetlenie ID osoby (nad prostokątem/kropką, przesunięte w górę o 20 pikseli)
            cv2.putText(img, f'ID: {track.track_id}', (cx, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Tworzenie folderu dla osoby
            person_folder = os.path.join(output_folder, f"person_{track.track_id}")
            os.makedirs(person_folder, exist_ok=True)

            # Zapisywanie zdjęcia osoby
            frame_count = len(os.listdir(person_folder)) + 1
            cv2.imwrite(os.path.join(person_folder, f"person_{track.track_id}_{frame_count}.jpg"), img_clean[y1:y2, x1:x2])

    # Wyświetlanie liczby osób na bieżącej klatce oraz liczby unikalnych osób
    cv2.putText(img, f'Current People: {frame_person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Unique People: {len(unique_person_ids)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Obliczanie i wyświetlanie FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Zapisanie klatki do wideo
    out.write(img)

    # Wyświetlanie obrazu
    cv2.imshow("Image with Segmentation", img)

    # Zatrzymywanie po naciśnięciu ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
