import os
import cv2

# Diretorios de input e output
input_dir  = 'videos'
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue

    video_path = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Erro ao abrir {filename}')
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gera 10 índices igualmente espaçados
    frame_ids = [round(i * (total_frames - 1) / 9) for i in range(10)]

    # Cria pasta específica para o vídeo
    name, _ = os.path.splitext(filename)
    video_out = os.path.join(output_dir, name)
    os.makedirs(video_out, exist_ok=True)

    for idx, fid in enumerate(frame_ids, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f' >>Falha ao ler frame #{fid} ({idx}/10)') # Caso aconteça algum problema
            continue

        frame_name = f'frame_{idx:02d}.jpg'
        cv2.imwrite(os.path.join(video_out, frame_name), frame)
        print(f'..Salvo {frame_name} em {name}/')

    cap.release()