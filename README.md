# Dependencies
- [mmpretrain](https://github.com/open-mmlab/mmpretrain)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- PyTorch=1.12.1
- CUDA=11.3

Installing MMdetection from source, then overwrite the original files using the files in the mmdetection folder in the repository.

# Data
- RUOD
- UIEB
- LSUI

# Train
OD Task: Refer to [mmdetection](https://github.com/open-mmlab/mmdetection)

UIE Task:
1. add **data_path** to **config.yml**
2. python **train.py**

# Test
OD Task: Refer to [mmdetection](https://github.com/open-mmlab/mmdetection)

UIE Task: Using [pyiqa](https://github.com/chaofengc/IQA-PyTorch) as the test-tool.

# Acknowledgements
The Code is based on [mmdetection](https://github.com/open-mmlab/mmdetection), [mmpretrain](https://github.com/open-mmlab/mmpretrain) and [lqit](https://github.com/BIGWangYuDong/lqit) and thanks to these repositories.