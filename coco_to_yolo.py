from ultralytics.data.converter import convert_coco

# Конвертация аннотаций из формата COCO в формат YOLO
convert_coco(
    labels_dir="Datasets/SeaDronesSees/annotations", 
    save_dir="Datasets/SeaDronesSees", 
    use_segments=False,    # Отключение конвертации масок (решается задача детекции)
    use_keypoints=False,   # Отключение конвертации ключевых точек
    cls91to80=False        # Сохранение оригинальных ID классов датасета
)