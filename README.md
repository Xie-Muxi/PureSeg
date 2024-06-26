# Getting Start

Experimental remote sensing segmentation project based on mmengine-template

# Installation
### Please install Pytorch2.x, mmcv2.0 firstly

```shell
git clone https://github.com/Xie-Muxi/PureSeg.git 
cd PureSeg
pip install -e .

# optiona
ln -s ${data_path} ./data # We recommend using symbolic link data to "./data"

```


### --------------- MMEngine-Template ---------------
## Introduction

This branch is a refactored version of [PyTorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), based on the **mmengine-template**. I would like to express my gratitude to the creators of PyTorch_Retinaface for their excellent work on implementing RetinaFace using PyTorch. Their project has been immensely helpful for developers interested in studying and applying RetinaFace.

In the field of deep learning, new training techniques and mature training pipelines are constantly being developed, which have significantly improved the efficiency of training. OpenMMLab's new generation training framework, [**MMEngine**](https://github.com/open-mmlab/mmengine), offers a range of out-of-the-box tools and high-performance training pipelines. Its standardized training process has greatly enhanced the scalability of projects. With this branch, RetinaFace has been refactored with MMEngine to provide users with a more efficient training experience.

## Installation

1. Follow the [official guide](https://pytorch.org/get-started/locally/) to install PyTorch.
2. Install requirements
   ```bash
   pip install -r requirements.txt
   ```

## Training/Testing/Inference

### Prepare dataset

1. Download [dataset](https://pan.baidu.com/s/15A9TGQvqqWIKr8OJnuRoJg?pwd=fexq); Access Code：`fexq`
2. extract the dataset to `data/widerface`

```bash
pip install -r requirements.txt
```

### Setup PYTHONPATH

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Training

1. Single GPU training

```bash
python tools/train.py configs/retinaface.py --work-dir work_dirs/retinanet --cfg-options train_dataloader.batch_size=24 --amp
```

2. MultiGPU training

```bash
 ./tools/dist_train.sh configs/retinaface.py 4 --work-dir ./work_dirs/retinanet --amp
```

### Testing

1. Single GPU testing

```bash
 python tools/test.py configs/retinaface.py work_dirs/retinanet/epoch_100.pth
```

2. MultiGPU testing

```bash
./tools/dist_test.sh configs/retinaface.py work_dirs/retinanet/epoch_100.pth 4
```

### Inference

```bash
python demo/mmengine_template_demo.py demo/test.jpg configs/retinaface.py work_dirs/retinanet/epoch_100.pth
```

## What is mmengine-template?

**MMEngine Template** is a template for the best practices in MMEngine. Config&Resgitry in [MMEngine](https://github.com/open-mmlab/mmengine) provides an elegant solution for developers to customize their modules and record experimental information. However, due to the learning curve of the Config&Registry system, developers often encounter some problems during development. Therefore, we provide the **MMEngine Template** to guide developers in achieving the best practices in various deep learning fields based on MMEngine.

You may wonder why there is a need for **MMEngine Template** since the downstream repositories (MMDet, MMCls) of OpenMMLab have already provided a "template". Actually, **MMEngine Template** provides a more lightweight development standard compared to the downstream algorithm repositories. **MMEngine Template** is designed to provide a generic development template to simplify the development process based on MMEngine.

Compared with downstream repositories, **MMEngine Template**:

- A simpler directory structure for easier maintenance. Since most developers do not need to customize multiple datasets, hooks, loops, etc., creating excessive nested directories is unnecessary.
- A more flexible data flow format. The data flow standards in OpenMMLab series repositories are quite strict and require compliance with `mmengine.structures.BaseDataElement`, which is not necessary for most individual developers. As a result, the MMEngine Template relaxes these standards and does not require developers to format their data as `mmengine.structures.BaseDataElement` instances in the data flow.

Developers often encounter the "Unregistered module xxx" error when they fail to trigger registration. To prevent this issue, the MMEngine Template automatically registers the module if developers follow the [default directory structure](#directory-structure).

## Directory structure

```bash
├── configs                                 Commonly used base config file.
├── demo
│   ├── mmengine_template_demo.py           General demo script
├── rsseg
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── datasets.py                     Customize your dataset here
│   │   └── transforms.py                   Customize your data transform here
│   ├── engine
│   │   ├── __init__.py
│   │   ├── hooks.py                        Customize your hooks here
│   │   ├── optimizers.py                   Less commonly used. Customize your optimizer here
│   │   ├── optim_wrappers.py               Less commonly used. Customize your optimizer wrapper here
│   │   ├── optim_wrapper_constructors.py   Less commonly used. Customize your optimizer wrapper constructor here
│   │   └── schedulers.py                   Customize your lr/momentum scheduler here
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py                    Less commonly used. Customize your evaluator here
│   │   └── metrics.py                      Customize your metric here.
│   ├── infer
│   │   ├── inference.py                    Used for demo script. Customize your inferencer here
│   │   └── __init__.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── model.py                        Customize your model here.
│   │   ├── weight_init.py                  Less commonly used here. Customize your initializer here.
│   │   └── wrappers.py                     Less commonly used here. Customize your wrapper here.
│   ├── registry.py
│   └── version.py
├── tools                                   General train/test script
```

## How to use

Assuming you have already understood the basic process of developing based on MMEngine, then when developing based on **MMEngine Template**, you need to customize your module in `datasets/datasets.py`, `datasets/transform.py`, `models/models.py`, etc., and training pipelines will be automatically built in `mmengine.Runner`. **MMEngine Template** provides a general training/testing/inferring script in `tools` and `demo`, and you can directly use them in the command line.

Besides, there are lots of `mmengine-template` or `mmengine_template` in this project, including file name, module name and scope name, you need to replace them with your own project name before organizing your code.

For advanced users, you may need to customize more components and register more modules. When developing, remember to update the locations parameter in registry.py when adding new modules to ensure that the newly added modules are correctly registered.

**MMEngine Template** will continuously update new branches to support various deep-learning tasks in different fields, so stay tuned.

## Acknowledgements

- [PyTorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
