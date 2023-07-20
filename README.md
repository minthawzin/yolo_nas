## YOLO NAS OBJECT DETECTION 

This repository contains the training script as well as inference script using YOLO_NAS model architecture.

### Setting up the environment

The repository is currently implemented using docker environment. Dockerfile is also provided in the repository for easier replication.

```
git clone https://github.com/minthawzin/yolo_nas.git
cd yolo_nas
docker build -t yolo_nas .
```

### Training on custom dataset

- Dataset Structure

The current dataset format is the same as YOLOv5 data format that is used for training purposes.

```
├── train
│   ├── images
│   │   ├── 273271,1a02900084ed5ae8.jpg
│   │   └── 273271,1b9eb00089049cd6.jpg
│   └── labels
│       ├── 273271,1a02900084ed5ae8.txt
│       └── 273271,1b9eb00089049cd6.txt
└── valid
    ├── images
    │   ├── 273271,1a02900084ed5ae8.jpg
    │   └── 273271,1b9eb00089049cd6.jpg
    └── labels
        ├── 273271,1a02900084ed5ae8.txt
        └── 273271,1b9eb00089049cd6.txt
```

After preparing the dataset, the training script can be readily run for training the model

```
python3 train.py
```


### Inference on image

```
python3 inference.py
```

