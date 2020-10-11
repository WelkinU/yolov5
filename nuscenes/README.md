# Train YOLOv5 On [NuImages Dataset](https://www.nuscenes.org/nuimages)

## Download and Prepare Data
1. Download dataset here: [https://www.nuscenes.org/download](https://www.nuscenes.org/download). Only need to download the metadata and samples (not the sweeps)
![](https://user-images.githubusercontent.com/47000850/92659515-7d6d8f80-f2c6-11ea-9be9-1002ae559dbc.png)
1. Move the `samples` and metadata folders into the `yolov5/nuscenes` folder, it should look like this:
![](https://user-images.githubusercontent.com/47000850/92659740-fa990480-f2c6-11ea-822e-c4c3bf3786eb.png)
1. To modify dataset class names or filter out dataset classes, modify the `class_map` variable in `data_prep.py`
1. Run `data_prep.py`. This will put the data in the format and file structure required by the repo (as defined by [#12](https://github.com/ultralytics/yolov5/issues/12)). It will create label files, images, and the `nuimages.yaml` file in the proper structures and overwrite existing files with same name.
1. When this completes (can take some time as the dataset is pretty large), you should end up with images in `yolov5/nuscenes/images/train` and `yolov5/nuscenes/images/val` and corresponding labels in the `yolov5/nuscenes/labels/train` and `yolov5/nuscenes/images/val` folders.

## Training
### Minimal Example - Train YOLOv5x model at max image size, single GPU

Run `python train.py --weights 'yolov5x.pt' --cfg ./models/yolov5x.yaml --data ./nuscenes/nuimages.yaml --img-size 1600 --batch-size 1 --epochs 1 --device '0'`

Notes:
* YOLOv5x at img-size=1600 batch_size=1 uses ~6GB gpu memory, but the torch NMS can spike much higher (>8GB). Can use `--notest` to avoid this.
* YOLOv5 uses a virtual batch size of 64, so gradient updates will be accumulated until hitting 64.
* Using the `--rect` argument disables mosaic augmentation. According to [#55](https://github.com/ultralytics/yolov5/issues/55) rect is faster and uses less memory, but mosaic augmentation is important for detecting smaller objects.

## Inference
Follow the instructions in the main repo to run inference using `detect.py`.
