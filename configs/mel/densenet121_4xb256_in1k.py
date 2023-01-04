_base_ = [
    '../densenet/densenet121_4xb256_in1k.py'
]

train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'pretrained_models/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone'
        )
    ),
    head=dict(
        num_classes=2,
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
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='Accuracy', topk=(1, )),
    dict(type='SingleLabelMetric', num_classes = 2),
]

test_evaluator = val_evaluator