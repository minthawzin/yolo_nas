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
|root_dataset_dir 
|_____train
|__________images
|_______________1.jpg
|_______________2.jpg
|__________labels
|_______________1.txt
|_______________2.txt
|_____valid
|___________...
```

After preparing the dataset, the training script can be readily run for training the model

```
python3 train.py
```


### Inference on image

```
python3 inference.py
```

