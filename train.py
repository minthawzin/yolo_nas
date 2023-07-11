from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models

from src.dataset.dataloader import DatasetLoader
from src.hyps.training_params import train_params

#trainer = Trainer(experiment_name="yolo_nas_s_soccer_players", ckpt_root_dir="./content/sg_checkpoints_dir/")
#net = models.get(Models.YOLO_NAS_S, num_classes=4, pretrained_weights="coco")

def main():

    CLASS_NAME = ['Background', 'Human']
    # Dataset Loading #
    dataloader = DatasetLoader( data_dir='./dataset/', class_names=CLASS_NAME, input_img=640 )
    dataloader.visualiseOriginalDataset( dataloader.train_dataset )

    # Model Initialiser *
    trainer = Trainer(experiment_name="crowdhuman_dataset", ckpt_root_dir="./content/sg_checkpoints_dir/")
    net     = models.get("yolo_nas_s", num_classes=len( CLASS_NAME )+1  , pretrained_weights='coco' )
    trainer.train(model=net, training_params=train_params, train_loader=dataloader.train_dataloader, valid_loader=dataloader.valid_dataloader)

if __name__ == "__main__":
    main()