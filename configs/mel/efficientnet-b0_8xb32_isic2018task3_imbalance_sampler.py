_base_ = [
    '../efficientnet/efficientnet-b0_8xb32_in1k.py'
]

# optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


model = dict(
    head=dict(
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, )))

dataset_type = 'CustomDataset'
classes = ['nv', 'mel']  # 数据集中各类别的名称


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
    train=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_GroundTruth/train_meta.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_GroundTruth/val_meta.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_Input',
        ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Validation_GroundTruth/val_meta.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)


# 构造训练集 dataloader
# train_dataloader = dict(
#     dataset=dict(                      # 训练数据集
#         type=dataset_type,
#         data_root='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_Input',
#         ann_file='/home/fate/gyj/datasets/ISIC2018/ISIC2018_Task3_Training_GroundTruth/train_meta.txt',
#         data_prefix='',
#         pipeline=train_pipeline),
#     sampler=dict(type='ImbalancedDatasetSampler', shuffle=True),   # 默认采样器
#     persistent_workers=True,                             # 是否保持进程，可以缩短每个epoch的准备时间
# )

sampler = dict(
    type='ImbalancedDatasetSampler'
)


load_from = 'pretrained_models/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth'



runner = dict(type='EpochBasedRunner', max_epochs=80)

evaluation = dict(interval=5, metric='accuracy',  metric_options={'topk': 1})

checkpoint_config = dict(interval=5)

log_config = dict(
    interval=100,                      # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),          # 用于记录训练过程的文本记录器(logger)。
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
)


