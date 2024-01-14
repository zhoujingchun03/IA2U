resnet_cfg = dict(
    backbone=dict(
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=512,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=10,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')