_base_ = [
    '../resnest/resnest101_32xb64_in1k.py'
]

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrained_models/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        num_classes=2,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=2,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, ),
    ),
)
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(300, 225)),
    dict(type='EfficientNetRandomCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Rotate', angle = 30),
    dict(type='PackClsInputs'),
    
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(300, 225)),
    dict(type='EfficientNetRandomCrop', scale=224),
    dict(type='PackClsInputs'),
]
    
dataset_type = 'CustomDataset'


data_root = '/home/fate/gyj/datasets/ISIC2018/'
train_dataloader = dict(
    batch_size= 16,
    dataset=dict(
        type= dataset_type,
        data_root= data_root,
        ann_file='ISIC2018_Task3_Training_GroundTruth/train_meta.txt',
        data_prefix='ISIC2018_Task3_Training_Input',
        classes=['nv', 'mel'],
        pipeline = train_pipeline
    )
)

val_dataloader = dict(
    batch_size= 16,
    dataset=dict(
        type= dataset_type,
        data_root= data_root,
        ann_file='ISIC2018_Task3_Validation_GroundTruth/val_meta.txt',
        data_prefix='ISIC2018_Task3_Validation_Input',
        classes=['nv', 'mel'],
        pipeline = test_pipeline
    )
)

val_cfg = dict()
test_cfg = dict()
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='Accuracy', topk=(1, )),
    dict(type='SingleLabelMetric', num_classes = 2),
]

test_evaluator = val_evaluator



data_preprocessor = dict(
    num_classes=2,
)

