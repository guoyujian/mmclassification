本文件夹存放的是各种模型在asan上训练的结果，使用的预训练模型基于imageNet1k

- experiment1: densenet121 load_from ×
- experiment2: efficientNet-b0 load_from ×
- experiment3: resnet50 load_from ×
- experiment4: resnext50 load_from ×
- experiment5: swin-tiny load_from ×
- experiment6: van-b1 load_from ×
- experiment7: Pretrained = None 的efficientnet-b0
- experiment8: Pretrained = True 且 imbalancedsampler的efficientnet-b0
- experiment9: 废弃；pretrained=isic，finetune线性层？
- experiment10: pretrained=None densenet121
- experiment11: pretrained=None resnet50
- experiment12: pretrained=None resnext50
- experiment13: efficientNet-b0 采用init_cfg的形式加载预训练模型
- experiment14: resnet50 采用init_cfg的形式加载预训练模型
- experiment15: densenet 采用init_cfg的形式加载预训练模型
- experiment16: resnext 采用init_cfg的形式加载预训练模型
- experiment17: resnet 采用init_cfg的形式加载预训练模型，预训练模型由dermnet得到。  基本是失败了
- experiment18: Pretrained = None 且 imbalancedsampler的resnext
- experiment19: Pretrained = True 且 focal loss的efficientnet-b0
- experiment20: Pretrained = None 且 focal loss的 resnext
- experiment21: pretrained=None resnet101
- experiment22: pretrained=None 且 focal loss的 resnet50
- experiment23: pretrained=None 且 imbalancedsampler的 resnet50
- experiment24: pretrained=True resnet101
- experiment25: pretrained=True eff-b0  LabelSmoothLoss
- experiment32: pretrained=True van CrossEntropyLoss 
- processed_pics: 数据处理流程
- expose_confusion_matrix_and_metrics.ipynb: 计算各项指标。