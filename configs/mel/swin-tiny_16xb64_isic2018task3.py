_base_ = [

    '../swin_transformer/swin-tiny_16xb64_in1k.py'
]


model = dict(

    head=dict(
        type='LinearClsHead',
        num_classes=2,
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5)
    ]))


dataset_type = 'CustomDataset'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(225, 300)),
    dict(type='RandomCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    # dict(type='Rotate', angle = 30),
    dict(type='Albu', transforms = [
        dict(
            type='RandomRotate90',
            p=0.5
        ),
        dict(
            type='Rotate',
            limit=30,
            p=0.5
        ),

    ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224)),
    # dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]


data = dict(
    # samples_per_gpu=64,
    # workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_GroundTruth/train_meta.txt',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_GroundTruth/val_meta.txt',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_GroundTruth/val_meta.txt',
        pipeline=test_pipeline
    )
)



load_from = 'pretrained_models/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'

runner = dict(type='EpochBasedRunner', max_epochs=80)

evaluation = dict(interval=5, metric='accuracy',  metric_options={'topk': 1})

checkpoint_config = dict(interval=5)

log_config = dict(
    # interval=100,                      # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),          # 用于记录训练过程的文本记录器(logger)。
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
)