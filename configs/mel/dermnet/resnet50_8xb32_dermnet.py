_base_ = [
    '../../resnet/resnet18_8xb32_in1k.py'
]

model = dict(
    head=dict(
        num_classes=23,
        topk=(1, 5)))


dataset_type = 'CustomDataset'
classes = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']  # 数据集中各类别的名称

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(224, 224), interpolation="bilinear"),
    # dict(type='Rotate', angle = 30),
    dict(type='Albu', transforms = [
        dict(
            # 短边resize到max_size
            type='SmallestMaxSize',
            max_size=512
        ),
        dict(
            # 短边resize到max_size
            type='RandomSizedCrop',
            min_max_height = (384, 512),
            height = 224,
            width = 224
        ),
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
        _delete_ = True,
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/dermnet/train',
        pipeline=train_pipeline
    ),
    val=dict(
        _delete_ = True,
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/dermnet/test',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        _delete_ = True,
        type=dataset_type,
        data_prefix='/home/fate/gyj/datasets/dermnet/test',
        classes=classes,
        pipeline=test_pipeline
    )

    
)


runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=5)

log_config = dict(
    interval=50,                      # 打印日志的间隔， 单位 iters
    hooks=[
        dict(type='TextLoggerHook'),          # 用于记录训练过程的文本记录器(logger)。
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
)


# load_from = 'pretrained_models/resnet50_8xb32_in1k_20210831-ea4938fc.pth'


# evaluation = dict(interval=1, metric=['accuracy', 'recall'],  metric_options={'topk': 1})

evaluation = dict(interval=5, metric='accuracy',  metric_options={'topk': 1}, save_best='auto')






