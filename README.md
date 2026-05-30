# SeaDronesSee Object Detection Project

Проект детекции объектов на морских изображениях с использованием YOLO (You Only Look Once) нейросетей. Разработан для обнаружения людей в воде, судов и других объектов с борта БПЛА.

## 📋 Описание

Проект реализует полный цикл работы с детекцией объектов:
- Конвертация аннотаций из формата COCO в YOLO
- Обучение модели YOLO на датасете SeaDronesSee
- Запуск предсказаний на изображениях и видео
- Потоковая обработка видео с многопоточной архитектурой

## 🎯 Классы объектов

Модель обучена обнаруживать 4 класса объектов:

| ID | Класс (EN) | Описание |
|----|------------|----------|
| 0 | swimmer | Человек в воде |
| 1 | big vessel | Крупные суда |
| 2 | small vessel | Малые суда |
| 3 | ignored | Игнорируемые объекты (буи, навигационные знаки) |

## ⚙️ Установка

### Установка зависимостей

```bash
pip install ultralytics opencv-python numpy torch torchvision matplotlib tqdm
```

## 🚀 Быстрый старт

### 1. Конвертация аннотаций

```bash
# Используя coco_to_yolo.py
python coco_to_yolo.py

# Или через CLI ultralytics
yolo detect convert label-format=coco format=yolo path=Datasets/SeaDronesSees/annotations/instances_train.json save_dir=Datasets/SeaDronesSees/labels/train
yolo detect convert label-format=coco format=yolo path=Datasets/SeaDronesSees/annotations/instances_val.json save_dir=Datasets/SeaDronesSees/labels/val
```

### 2. Обучение модели

Запустить обучение модели:

```bash
python train_yolo.py
```

Параметры обучения (настраиваются в `train_yolo.py`):
- `model_name`: 'yolo26n'
- `epochs`: 16
- `imgsz`: 640
- `batch`: 48
- `device`: 0 (GPU)

Результаты обучения сохраняются в папку `seadrone_yolo/`.

### 3. Запуск предсказаний

#### На изображениях и видео:

```bash
python run_prediction.py
```

#### Потоковая обработка видео (многопоточная):

```bash
python run_prediction_stream.py
```

## 📊 Результаты обучения

После обучения модель сохраняет лучшие веса в:
```
seadrone_yolo/runs/detect/yolo26n_seadrone/weights/best.pt
```

## 🎥 Потоковая обработка видео

Скрипт `run_prediction_stream.py` реализует многопоточную архитектуру для обработки видео в реальном времени:

- **Основной поток**: чтение кадров из видеофайла
- **Поток инференса**: выполнение предсказаний модели
- **Очереди**: синхронизация между потоками с минимальной задержкой

Особенности:
- Порог уверенности: 0.35
- Цветовая схема для классов:
  - 🟥 Красный: swimmer (человек в воде)
  - 🟩 Зеленый: big vessel (крупные суда)
  - 🟦 Синий: small vessel (малые суда)
  - ⬜ Серый: ignored (игнорируемые объекты)

## 📚 Источники

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [SeaDronesSee Dataset](https://seadronessee.cs.uni-tuebingen.de/dataset)