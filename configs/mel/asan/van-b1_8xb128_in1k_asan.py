_base_ = [
    '../../van/van-b1_8xb128_in1k.py'

]


classes = [
    'nv',
    'mel'
]
model = dict(
    backbone = dict(
        # frozen_stage = 2,
        init_cfg = dict(
            type = 'Pretrained',
            checkpoint = 'pretrained_models/van-small_8xb128_in1k_20220501-17bc91aa.pth',
            prefix = 'backbone'
        )
    ),
    head=dict(
        num_classes=2,
        loss=dict(_delete_ = True, type='CrossEntropyLoss', loss_weight=1.0),
    ),
    # train_cfg=dict(
    #     _delete_ = True,
    #     augments=[
    #         # dict(type='BatchMixup', alpha=0.8, num_classes=2, prob=0.5), # default prob = 0.5
    #         # dict(type='BatchCutMix', alpha=1.0, num_classes=2, prob=0.5) # default prob = 0.5
    #     ]
    # )
    train_cfg = None
)


dataset_type = 'CustomDataset'



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


# load_from = 'pretrained_models/van-small_8xb128_in1k_20220501-17bc91aa.pth'

runner = dict(type='EpochBasedRunner', max_epochs=200)

evaluation = dict(interval=5, metric='accuracy',  metric_options={'topk': 1}, save_best='auto')

checkpoint_config = dict(interval=5)

log_config = dict(
    interval=20,                      # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),          # 用于记录训练过程的文本记录器(logger)。
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
)

