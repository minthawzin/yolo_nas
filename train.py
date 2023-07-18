from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from src.dataset.dataloader import DatasetLoader
from src.hyps.training_params import getTrainingParams

CLASS_NAME           = ['Human']
INPUT_SIZE           = 640
YOLO_NAS_MODEL_VER   = "YOLO_NAS_S" # M -> MEDIUM, L -> LARGE , ETC... 

def loadDataset( input_root_dir, class_name ):
    dataloader = DatasetLoader( data_dir=input_root_dir, class_names=class_name, input_img=INPUT_SIZE )
    dataloader.visualiseOriginalDataset( dataloader.train_dataloader )
    dataloader.visualiseAugmentedDataset( dataloader.train_dataloader )

    return dataloader

def main():

    # Dataset Loading #
    dataloader = loadDataset( './datasets/', CLASS_NAME )
  
    # Model Initialiser #
    trainer = Trainer(experiment_name="crowdhuman_dataset", ckpt_root_dir="./content/sg_checkpoints_dir/")
    net     = models.get(  YOLO_NAS_MODEL_VER, num_classes=len( CLASS_NAME ) )

    # train parameters #
    train_params = getTrainingParams( CLASS_NAME )
    trainer.train(model=net, training_params=train_params, train_loader=dataloader.train_dataloader, valid_loader=dataloader.valid_dataloader)

if __name__ == "__main__":
    main()