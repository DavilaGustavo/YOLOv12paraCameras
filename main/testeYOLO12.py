import os
import cv2
from ultralytics import YOLO

# Diretorios de input, output e modelo
frames_dir    = 'frames'
results_dir   = 'resultados'
weights_path  = 'yolo12m.pt'    # yolo12n.pt yolo12s.pt yolo12m.pt yolo12l.pt yolo12x.pt
os.makedirs(results_dir, exist_ok=True)

# Carrega o modelo
model = YOLO(weights_path)
# results = model.train(data="lvis.yaml", epochs=100, imgsz=640) # INVIAVEL SEM GPU

# Itera sobre cada pasta em frames
for video_name in os.listdir(frames_dir):
    video_frames_dir = os.path.join(frames_dir, video_name)
    if not os.path.isdir(video_frames_dir):
        continue

    # Cria pasta de saída para este vídeo
    video_results_dir = os.path.join(results_dir, video_name)
    os.makedirs(video_results_dir, exist_ok=True)

    # Para cada frame na pasta
    for frame_file in sorted(os.listdir(video_frames_dir)):
        frame_path = os.path.join(video_frames_dir, frame_file)
        img = cv2.imread(frame_path)

        # Inferência
        results = model(img)[0]

        # Desenha detection box
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

            # Gambiarra para visualizar em qualquer tom
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Salva a imagem
        out_path = os.path.join(video_results_dir, frame_file)
        cv2.imwrite(out_path, img)
        print(f'CONCLUIDO: {video_name}/{frame_file}')