# Train YOLOX using the COCO dataset

This floder showcases how to use BigDL Nano to accelerate [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) model training. [COCO](https://cocodataset.org/#home) dataset is used in this example.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments.
```bash
git clone https://github.com/y199387/YOLOX.git
conda create -n bigdl python==3.7.10 setuptools==58.0.4
conda activate bigdl
pip install --pre --upgrade bigdl-nano[pytorch]
source bigdl-nano-init
cd YOLOX
pip install -v -e .
```

## Prepare the data
Download COCO 2017 images from [here](http://images.cocodataset.org/zips/train2017.zip) and 2017 annotations from [here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
```bash
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

## Train YOLOX
### Training with single process, example command:
```bash
python -m yolox.tools.train -n yolox_s -b 64 --use_ipex --precision 32
                               yolox-m                              bf16
                               yolox-l                              64
                               yolox-x                              16
```
- -b: total batch size
- --use_ipex: enable IPEX acceleration
- --precision: mixed precision for training
### Training with multi-instance, example command
```bash
python -m yolox.tools.train -n yolox_s -b 64 --num_processes 2 --strategy subprocess
                               yolox-m
                               yolox-l
                               yolox-x
```
- --num_processes number of processes for training
- --strategy distributed backend