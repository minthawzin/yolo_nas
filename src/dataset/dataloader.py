from super_gradients.training.datasets.detection_datasets.yolo_format_detection import YoloDarknetFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training import dataloaders
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.transforms.transforms import DetectionTransform, DetectionTargetsFormatTransform, DetectionTargetsFormat
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
import cv2, random, os
import numpy as np
import matplotlib.pyplot as plt

CLASS_COLORS = [(0,255,0),(0,0,255),(255,0,0)]

class DatasetLoader:

    def __init__( self, data_dir, class_names, input_img=640 ):
        self.data_dir         = data_dir
        self.class_names      = class_names 
        self.input_img_shape  = input_img
        self.batch_size       = 4

        self.train_dataset    = self.getYOLODataset( mode='train' )
        self.valid_dataset    = self.getYOLODataset( mode='valid' )

        self.train_dataloader = self.getParams( mode='train' )
        self.valid_dataloader = self.getParams( mode='valid')

    def getTransforms( self, mode='train' ):
        if mode == 'train':
            transforms=[
                #DetectionMosaic(prob=1., input_dim=( self.input_img_shape, self.input_img_shape)),
                DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                    target_size=( self.input_img_shape, self.input_img_shape),
                                    filter_box_candidates=False, border_value=128),
                DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                DetectionHorizontalFlip(prob=0.5),
                DetectionPaddedRescale(input_dim=(self.input_img_shape, self.input_img_shape), max_targets=300),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(max_targets=300, input_dim=(self.input_img_shape, self.input_img_shape),
                                                output_format="LABEL_CXCYWH")
            ]
        else:
            transforms = [
                DetectionPaddedRescale(input_dim=(self.input_img_shape, self.input_img_shape), max_targets=300),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(max_targets=300, input_dim=(self.input_img_shape, self.input_img_shape),
                                                output_format="LABEL_CXCYWH")
            ]

        return transforms
            
    def getYOLODataset( self, mode='train' ):

        yolo_dataset = YoloDarknetFormatDetectionDataset(
                            data_dir                 = f"{self.data_dir}",
                            images_dir               = f"{mode}/images",
                            labels_dir               = f"{mode}/labels",
                            input_dim                = ( self.input_img_shape, self.input_img_shape ),
                            classes                  = self.class_names,
                            ignore_empty_annotations = False,
                            transforms               = self.getTransforms( mode=mode )
        )

        return yolo_dataset

    def visualiseOriginalDataset( self, dataset, num_plots=3, num_images=9, img_per_row=3, output_dir="./output/visualise/"):

        os.makedirs( output_dir, exist_ok=True )

        target_transformFormat = DetectionTargetsFormatTransform( input_format=dataset.original_target_format, output_format=XYXY_LABEL )

        for plot_i in range( num_plots ):
            fig = plt.figure( figsize=(10,10) )
            n_subplot = int( np.ceil( num_images ** 0.5))

            image_row_list = []
            overall_image  = []
            for img_i, _ in enumerate(range( num_images+1 )):
                random_index = random.randint(0, dataset.n_available_samples )
                sample = dataset.get_sample( random_index )
                image, targets = sample['image'], sample['target']
                if img_i % img_per_row != 0 or img_i == 0:
                    for target in targets:
                        x1, y1, x2, y2, class_index = target
                        cv2.rectangle( image, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS[int(class_index)], thickness=2 )
                    image = cv2.resize( image, ( self.input_img_shape, int(self.input_img_shape/2) ))
                    image_row_list.append( image )
                else:
                    
                    combined_image  = cv2.hconcat( image_row_list )
                    overall_image.append( combined_image )
                    image_row_list = []
                    for target in targets:
                        x1, y1, x2, y2, class_index = target
                        cv2.rectangle( image, (int(x1), int(y1)), (int(x2), int(y2)), CLASS_COLORS[int(class_index)], thickness=2 )
                    image = cv2.resize( image, ( self.input_img_shape, int(self.input_img_shape/2) ))
                    image_row_list.append( image )

            overall_image = cv2.vconcat( overall_image )
            output_path   = f"{output_dir}/{plot_i}_original.png"
            cv2.imwrite( output_path, overall_image )

    def getParams( self, mode='train' ):
        if mode=='train':
            data_loader = dataloaders.get(dataset= self.train_dataset, dataloader_params={
            "shuffle": True,
            "batch_size": self.batch_size,
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": DetectionCollateFN(),
            "worker_init_fn": worker_init_reset_seed,
            "min_samples": 512
        })
            
        else:
            data_loader = dataloaders.get(dataset=self.valid_dataset, dataloader_params={
            "shuffle": False,
            "batch_size": self.batch_size,
            "num_workers": 2,
            "drop_last": False,
            "pin_memory": True,
            "collate_fn": DetectionCollateFN(),
            "worker_init_fn": worker_init_reset_seed
        })
        return data_loader