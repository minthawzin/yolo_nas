from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

train_params = {
    "warmup_initial_lr": 1e-6,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "AdamW",
    "zero_weight_decay_on_bias_and_bn": True,
    "lr_warmup_epochs": 3,
    "warmup_mode": "linear_epoch_step",
    "optimizer_params": {"weight_decay": 0.0001},
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(use_static_assigner=False, num_classes=3, reg_max=16),
    "valid_metrics_list": [
        DetectionMetrics_050(score_thres=0.1, top_k_predictions=300, num_cls=2, normalize_targets=True,
                             post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                    nms_top_k=1000, max_predictions=300,
                                                                                    nms_threshold=0.7))],

    "metric_to_watch": 'mAP@0.50'}