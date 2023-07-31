import torch, cv2, os, time
from super_gradients.training import models
from src.dataset.image_utils import *
from src.file.file_loader import FileLoader

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
    loaded_model     = models.get( YOLO_NAS_MODEL_VER, num_classes=len(CLASS_NAMES) )
    loaded_model.to('cuda')
    loaded_model.eval()
    return loaded_model

# Input image 
def loadMedia( media_path ):
    if media_path.endswith('.png') or media_path.endswith('.jpg') or media_path.endswith('.jpeg'):
        input_image_path = media_path
        original_image   = cv2.imread( input_image_path )
        processed_image  = preprocessCV( original_image, INPUT_SIZE )
        input_type       = "image"
        return processed_image, input_type
    else:
        input_video_path = media_path
        input_video      = FileLoader( input_video_path ).movieLoader
        input_type       = "video"
        return input_video, input_type
    
# Predict results
def ImageInference( loaded_model, processed_image ):
    with torch.no_grad():
        
        model_output       = loaded_model.predict( processed_image, conf=CONFIDENCE_THRESHOLD )._images_prediction_lst[0]

        print( model_output )

        image              = model_output.image
        image              = cv2.cvtColor( image, cv2.COLOR_RGB2BGR )
        prediction_results = model_output.prediction
        
        print( "Inference Results" )
        print( "-----------------" )
        for box, class_index, confidence in zip( prediction_results.bboxes_xyxy, prediction_results.labels, prediction_results.confidence ):
            x1, y1, x2, y2 = map( int, box )
            class_name     = CLASS_NAMES[int(class_index)]
            confidence     = round( confidence, 2 )
            print(f"BBox: [{ x1, y1, x2, y2 }], Class: {class_name}, Confidence: {confidence}")
            cv2.rectangle( image , ( x1, y1 ), ( x2, y2 ), ( 0, 255, 0), 2 )

    return image

def VideoInference( loaded_model, video_loader ):

    start_time = time.time()
    for frame_id in range( video_loader.maxMovieFrame ):
        video_loader.readFrame()
        if video_loader.currentFrame is None:
            break
        processed_image = preprocessCV( video_loader.currentFrame, INPUT_SIZE )
        result_image    = ImageInference( loaded_model, processed_image )

        cv2.imshow('output', result_image )
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            os._exit(0)

    time_taken = time.time() - start_time
    fps        = video_loader.maxMovieFrame/time_taken
    print( f"FPS: {fps}" ) 

def main():

    input_model_path   = './weights/ckpt_best.pth'
    input_media_path   = './test_videos/test.jpeg'
    loaded_model       = loadModel( input_model_path )
    loaded_media, type = loadMedia( input_media_path )
    if type == 'video':
        VideoInference( loaded_model, loaded_media )
    else:
        result_image   = ImageInference( loaded_model, loaded_media )
        cv2.imshow('output', result_image )
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

