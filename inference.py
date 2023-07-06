import os
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.processing import ComposeProcessing

from src.dataset.dataloader import train_loader, valid_loader
from src.hyps.training_params import train_params


net = models.get(Models.YOLO_NAS_S, num_classes=3, checkpoint_path="./content/sg_checkpoints_dir/yolo_nas_s_soccer_players/ckpt_best.pth")
prediction = net.predict("http://www.runofplay.com/blog/wp-content/uploads/2011/02/nearside.jpg")

prediction.save('./output/')