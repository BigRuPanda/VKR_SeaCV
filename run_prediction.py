from ultralytics import YOLO
import cv2
import os

model = 'yolo26n'
project = 'seadrone_yolo'
video_path = 'videos/1.mp4'
image_path = 'Datasets/SeaDronesSees/images/val/71.jpg'
name = str(f'runs/runs_{model}')
show = True

def convert_avi_to_mp4_in_folder(root_folder):
    '''
    Рекурсивно конвертирует все AVI-файлы в MP4 в указанной папке и всех её подпапках.
    Пропускает файлы, если .mp4 уже существует.
    Создаёт аналогичную структуру папок для выходных файлов.
    '''
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.avi'):
                input_path = os.path.join(dirpath, filename)
                # Заменяем расширение .avi на .mp4
                output_filename = os.path.splitext(filename)[0] + '.mp4'
                output_path = os.path.join(dirpath, output_filename)
                
                # Проверяем, существует ли уже .mp4 файл
                if os.path.exists(output_path):
                    print(f'Пропущено (уже существует): {output_path}')
                    continue  # Пропускаем конвертацию
                
                print(f'Конвертация: {input_path} -> {output_path}')
                convert_single_avi_to_mp4(input_path, output_path)

def convert_single_avi_to_mp4(input_path, output_path):
    '''
    Конвертирует один AVI-файл в MP4 с помощью OpenCV.
    '''
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f'Ошибка: не удалось открыть файл {input_path}')
        return
    
    # Получаем свойства видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определяем кодек и создаём VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Читаем и записываем кадры
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    print(f'Видео сохранено как {output_path}')

if __name__ == '__main__':
    # Загрузка обученной модели
    model = YOLO(str(f'runs/detect/seadrone_yolo/{model}_seadrone/weights/best.pt'))

    # Обработка видео
    model.predict(video_path, show=show, save=True, project=project, name=name)

    # Обработка изображения
    model.predict('Datasets/SeaDronesSees/images/val/71.jpg', show=show, save=True, project=project, name=name)

    # Конвертация видео в mp4
    convert_avi_to_mp4_in_folder('seadrone_yolo/runs', name)
