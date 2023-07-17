import os
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.processing import ComposeProcessing


net = models.get(Models.YOLO_NAS_S, num_classes=3, checkpoint_path="./crowdhuman_dataset/ckpt_best.pth")
prediction = net.predict("http://www.runofplay.com/blog/wp-content/uploads/2011/02/nearside.jpg")

prediction.save('./output/')