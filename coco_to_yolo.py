from ultralytics.data.converter import convert_coco

# Convert COCO annotations to YOLO format
convert_coco(
    labels_dir="Datasets/SeaDronesSees/annotations",      # dir containing instances_train.json, etc.
    save_dir="Datasets/SeaDronesSees",                    # root output dir (creates labels/train/, labels/val/)
    use_segments=False,                                   # False for detection (not segmentation)
    use_keypoints=False,                                  # False unless pose
    cls91to80=False,                                      # True only for standard COCO 91→80; keep False for custom
    lvis=False
)
