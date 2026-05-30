import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO

# Параметры проекта
project = 'runs/detect/seadrone_yolo/yolo26n_seadrone'
model_path = f'{project}/weights/best.pt'
video_path = 'videos/1.mp4'

# Соответствие классов и цветов (согласно 4-классовой схеме из Главы 4)
LABELS_EN = {
    0: "swimmer",       # человек в воде
    1: "big vessel",    # крупные суда
    2: "small vessel",  # малые суда
    3: "ignored",       # игнорируемые объекты (буи, навигационные знаки)
}

COLORS = {
    0: (0, 0, 255),      # swimmer - красный (критический класс)
    1: (0, 255, 0),      # big vessel - зелёный
    2: (255, 0, 0),      # small vessel - синий
    3: (128, 128, 128),  # ignored - серый
}

# Порог уверенности для отображения (YOLO также фильтрует на уровне инференса)
CONFIDENCE_THRESHOLD = 0.35

# Очередь для кадров (максимум 1, чтобы не накапливать задержку и работать в real-time)
frame_queue = queue.Queue(maxsize=1)
# Очередь для результатов детекции
result_queue = queue.Queue(maxsize=1)

stop_event = threading.Event()

def inference_worker(model):
    """Поток для выполнения инференса нейросети, чтобы не блокировать чтение видео."""
    print("Поток обработки (Inference Thread) запущен")
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Покадровый инференс модели
        results = model(frame, verbose=False)
        
        try:
            # Извлекаем тензор боксов: [x1, y1, x2, y2, confidence, class_id]
            boxes_data = results[0].boxes.data.cpu().numpy()
            # Кладем в очередь. Если очередь полна (предыдущий кадр еще не отрисован),
            # пропускаем старый результат, чтобы минимизировать latency.
            try:
                result_queue.put_nowait(boxes_data)
            except queue.Full:
                pass
        except Exception as e:
            print(f"Ошибка обработки результатов: {e}")
        
        frame_queue.task_done()
    
    print("Поток обработки остановлен")

def main():
    try:
        model = YOLO(model_path)
        print(f"Модель {model_path} успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео '{video_path}'")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / video_fps) if video_fps > 0 else 30
    print(f"Оригинальный FPS: {video_fps:.2f} (Задержка: {wait_ms}ms)")

    # Запуск потока инференса
    worker_thread = threading.Thread(target=inference_worker, args=(model,), daemon=True)
    worker_thread.start()

    print("Симуляция стрима запущена. Нажмите 'q' для выхода")

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Видео завершено")
            break

        # Отправляем кадр в очередь на обработку
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass # Если поток инференса не успевает, пропускаем кадр для сохранения real-time

        # Получаем свежие результаты детекции
        try:
            latest_boxes_raw = result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            latest_boxes_raw = None

        # Отрисовка результатов
        display_frame = frame.copy()
        
        if latest_boxes_raw is not None and len(latest_boxes_raw) > 0:
            for box in latest_boxes_raw:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)
                
                # Дополнительная фильтрация по порогу уверенности
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                color = COLORS.get(cls_id, (255, 255, 255))
                label = f"{LABELS_EN.get(cls_id, f'class_{cls_id}')} {conf:.2f}"
                thickness = 2

                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(display_frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        cv2.imshow("Live Stream (YOLO26 Inference)", display_frame)

        # Поддержание FPS видео
        elapsed_ms = (time.time() - start_time) * 1000
        actual_wait = max(1, wait_ms - int(elapsed_ms))
        
        if cv2.waitKey(actual_wait) & 0xFF == ord('q'):
            break

    print("Остановка...")
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    worker_thread.join(timeout=2)
    print("Выход")

if __name__ == '__main__':
    main()
