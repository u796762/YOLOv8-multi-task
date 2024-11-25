import sys
import os.path as osp
from ultralytics import YOLO

repo_dir = osp.dirname(osp.abspath(__file__))

sys.path.insert(0, f"{repo_dir}/ultralytics")

# You should change the path to your local path to "ultralytics" file
model = YOLO(
    f'{repo_dir}/runs/multi/v4_6408/weights/best.pt',
    #f"{repo_dir}/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml",
    task="multi",
)
img_fp = f"{repo_dir}/datasets/DetSeg/yolo_datasets/toy_dataset_3430/images/val2017/b6f4b139-744f76c0.jpg"
# imgsz=(384,672), device=[3],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False)

metrics = model.predict(
    source=img_fp,
    device=[0],
    name='v4_daytime',
    save=True,
    conf=0.25, 
    iou=0.45, 
    # imgsz=(384,672),
    show_labels=True)
    # task='multi',
