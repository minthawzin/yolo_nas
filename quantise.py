from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from src.dataset.dataloader import DatasetLoader
from src.hyps.training_params import getTrainingParams

CLASS_NAME           = ['Human']
INPUT_SIZE           = 640
YOLO_NAS_MODEL_VER   = "YOLO_NAS_S" # M -> MEDIUM, L -> LARGE , ETC... 
DATASET_DIR          = "./mini_crowdhuman/"
DATASET_NAME         = 'MINI_CROWDHUMAN'
OUTPUT_DIR           = './content/sg_checkpoints_dir/'
TRAINED_MODEL_PATH   = './weights/ckpt_best.pth'
EPOCHS               = 10

def loadDataset( input_root_dir, class_name ):
    dataloader = DatasetLoader( data_dir=input_root_dir, class_names=class_name, input_img=INPUT_SIZE )
    dataloader.visualiseOriginalDataset( dataloader.train_dataloader )
    dataloader.visualiseAugmentedDataset( dataloader.train_dataloader )

    return dataloader

def main():

    # Dataset Loading #
    dataloader = loadDataset( DATASET_DIR, CLASS_NAME )
  
    # Model Initialiser #
    trainer = Trainer(experiment_name = DATASET_NAME, ckpt_root_dir = OUTPUT_DIR )
    net     = models.get(  YOLO_NAS_MODEL_VER, num_classes=len( CLASS_NAME ), checkpoint_path=TRAINED_MODEL_PATH )

    # train parameters #
    train_params = getTrainingParams( CLASS_NAME, EPOCHS )
    trainer.qat(model=net, training_params=train_params, train_loader=dataloader.train_dataloader, valid_loader=dataloader.valid_dataloader, calib_loader=dataloader.train_dataloader)

if __name__ == "__main__":
    main()