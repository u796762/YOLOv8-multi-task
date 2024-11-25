import sys
import os.path as osp
from ultralytics import YOLO

repo_dir = osp.dirname(osp.abspath(__file__))

sys.path.insert(0, f"{repo_dir}/ultralytics")

# You should change the path to your local path to "ultralytics" file
model = YOLO(
    f"{repo_dir}/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml",
    task="multi",
)
# You need to change the model path for yours.
# The model files saved under "./ultralytics/models/v8"
model.train(
    data=f"{repo_dir}/datasets/DetSeg/yolo_datasets/toy_dataset_3430/yolo_dataset.yaml",
    # f"{repo_dir}/ultralytics/datasets/bdd-multi-toy.yaml",
    batch=4,
    epochs=30,
    imgsz=(640, 640),
    device=[0],
    name="v4_640",
    val=True,
    task="multi",
    classes=[2, 3, 4, 9, 10, 11],
    combine_class=[2, 3, 4, 9],
    single_cls=True,
)
