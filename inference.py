import torch, cv2
from super_gradients.common.object_names import Models
from super_gradients.training import models
from src.dataset.image_utils import *

CLASS_NAMES          = ['Human']
INPUT_SIZE           = 640
CONFIDENCE_THRESHOLD = 0.5
YOLO_NAS_MODEL_VER   = "YOLO_NAS_S" # M -> MEDIUM, L -> LARGE , ETC... 

# Model Initialiser
def loadModel( model_path ):
    """
    Function load YOLONAS model
    """
    input_model_path = model_path
    loaded_model     = models.get( YOLO_NAS_MODEL_VER, num_classes=len(CLASS_NAMES)+1, checkpoint_path=input_model_path )
    loaded_model.to('cuda')
    loaded_model.eval()
    return loaded_model

# Input image 
def loadImage( image_path ):
    input_image_path = image_path
    original_image   = cv2.imread( input_image_path )
    processed_image  = preprocessCV( original_image, INPUT_SIZE )
    return processed_image

# Predict results
def inference( loaded_model, processed_image ):
    with torch.no_grad():
        model_output = next( loaded_model.predict( processed_image, conf=CONFIDENCE_THRESHOLD )._images_prediction_lst )
        image              = model_output.image
        image              = cv2.cvtColor( image, cv2.COLOR_RGB2BGR )
        prediction_results = model_output.prediction
        
        print( "Inference Results")
        print( "-----------------")
        for box, class_index, confidence in zip( prediction_results.bboxes_xyxy, prediction_results.labels, prediction_results.confidence ):
            x1, y1, x2, y2 = map( int, box )
            class_name     = CLASS_NAMES[int(class_index)]
            confidence     = round( confidence, 2 )
            print(f"BBox: [{ x1, y1, x2, y2 }], Class: {class_name}, Confidence: {confidence}")
            cv2.rectangle( image , ( x1, y1 ), ( x2, y2 ), ( 0, 255, 0), 2 )

        cv2.imshow('output', image )
        cv2.waitKey(0)

def main():

    input_model_path = './weights/average_model.pth'
    input_image_path = './test_videos/test.jpeg'
    loaded_model     = loadModel( input_model_path )
    loaded_image     = loadImage( input_image_path )
    inference( loaded_model, loaded_image )

if __name__ == "__main__":
    main()

