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

metrics = model.val(
    data=f"{repo_dir}/datasets/DetSeg/yolo_datasets/toy_dataset_24/yolo_dataset.yaml",
    device=[0],
    task='multi',
    name='val',
    iou=0.6,
    conf=0.001, 
    imgsz=(640,640),
    classes=[2,3,4,9,10,11],
    combine_class=[2,3,4,9],
    single_cls=True
    )

