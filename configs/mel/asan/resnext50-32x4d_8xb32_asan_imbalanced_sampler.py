_base_ = [
    '../../resnext/resnext50-32x4d_8xb32_in1k.py'
]


model = dict(
    head=dict(
        num_classes=2,
        topk=(1, ),
    )
)


dataset_type = 'CustomDataset'
classes = ['nv', 'mel']  # 数据集中各类别的名称


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224), interpolation="bilinear"),
    # dict(type='Rotate', angle = 30),
    dict(type='Albu', transforms = [
        dict(
            # 随机水平翻转
            type='VerticalFlip',
        ),
        dict(
            # 随机垂直翻转
            type='HorizontalFlip',
        ),
        dict(
            # 随机gamma变换
            type='RandomGamma',
        ),
        dict(
            # 随机亮度对比度
            type = 'RandomBrightnessContrast'
        ),
        dict(
            # 高斯噪声
            type = 'GaussNoise'
        ),
        dict(
            # 随机旋转0-90°，用黑色填充
            type='Rotate',
            border_mode=0,
        ),
    ]),
    dict(type='Normalize', **img_norm_cfg), # 归一化
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, 224), interpolation="bilinear"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    train=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/asan/images/train',
        ann_file='/home/fate/gyj/datasets/asan/asan_train_meta.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/asan/images/test',
        ann_file='/home/fate/gyj/datasets/asan/asan_test_meta.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/asan/images/test',
        ann_file='/home/fate/gyj/datasets/asan/asan_test_meta.txt',
        classes=classes,
        pipeline=test_pipeline
    )

    
)



# load_from = 'pretrained_models/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'



runner = dict(type='EpochBasedRunner', max_epochs=200)

evaluation = dict(interval=5, metric='accuracy',  metric_options={'topk': 1}, save_best='auto')

checkpoint_config = dict(interval=5)

log_config = dict(
    interval=50,                      # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),          # 用于记录训练过程的文本记录器(logger)。
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
)



sampler = dict(
    type='ImbalancedDatasetSampler'
)


