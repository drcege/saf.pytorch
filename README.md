## Segmentation-Aggregation Framework for Weakly Supervised Object Detection


### Contents
1. [Requirements: hardware](#requirements-hardware)
2. [Requirements: software](#requirements-software)
3. [Installation](#installation)
4. [Data preparation](#data-preparation)
5. [Testing](#testing)
6. [Training](#training)


### Requirements: hardware

- This code is GPU-only
- Tested on Ubuntu 18.04 with NVIDIA Tesla P100 and CUDA 9.0


### Requirements: software

- Implemented by Python 3.6
- Core dependencies are listed in [environment.yaml](https://github.com/SA-Framework/saf.pytorch/blob/master/environment.yaml)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/)
2. Create an environment based on the environment file
    ```shell
    conda env create -f environment.yaml
    ```
3. Activate this environment
   ```shell
   conda activate saf
   ```


### Installation

1. Clone the repository
    ```Shell
    git clone https://github.com/SA-Framework/saf.pytorch
    cd saf.pytorch
    export SAF_ROOT=`pwd`
    ```
2. Compile the CUDA code
    ```Shell
    cd $SAF_ROOT/libs
    sh make.sh
    ```


### Data preparation

1. Create data folder
    ```shell
    mkdir -p $SAF_ROOT/data/VOC2007/
    cd $SAF_ROOT/data/VOC2007/
    ```
2. Download the training, validation, test data and VOCdevkit of [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
    ```Shell
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```
3. Extract all these tars into one directory named `VOCdevkit`
    ```Shell
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_18-May-2011.tar
    ```
4. Download the COCO format pascal annotations from [here](https://drive.google.com/open?id=1lkSho3e6WDuKovZIVs6TK_YE7BOQ7rOe) and put them into the `annotations` directory
5. Create symlinks to image files
    ```shell
    ln -s VOCdevkit/VOC2007/JPEGImages/
    ```
6. The directory structure should look like this
    ```Shell
    $SAF_ROOT/data/VOC2007/
    $SAF_ROOT/data/VOC2007/annotations
    $SAF_ROOT/data/VOC2007/JPEGImages -> VOCdevkit/VOC2007/JPEGImages/
    $SAF_ROOT/data/VOC2007/VOCdevkit        
    ```
7. [*Optional*] Follow similar steps to get [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
8. Download and put the [precomputed proposals](https://drive.google.com/drive/folders/1YIYA8wwUMwJyGsY-TP3u20x0Jquk5kFL?usp=sharing) under `$SAF_ROOT/data/precomputed_proposals/`


### Testing

Our trained models are available [here](https://drive.google.com/open?id=1wXstUeiZXQDaMIls_ZsycjhjcsRrOQb4). Put them under `$SAF_ROOT/data/saf_models`.

#### On trainval set (CorLoc)

  ```shell
  python3 tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
    --set MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS False \
    --load_ckpt data/saf_models/voc2007.pth \
    --dataset voc2007trainval
  ```

#### On test set (mAP)
  ```Shell
  python3 tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
    --set MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS False \
    --load_ckpt data/saf_models/voc2007.pth \
    --dataset voc2007test
  ```
    
*Note: Add `--multi-gpu-testing` if multiple gpus are available.*


### Training 

  Download the backbone [VGG-16](https://drive.google.com/open?id=1wdZMGU5Vtna__AxsgDD8XHb6MhW0l5dg) model (pre-trained on ImageNet) and put it under `$SAF_ROOT/data/pretrained_model/`.
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python3 tools/train_net_step.py --dataset voc2007 \
    --cfg configs/baselines/vgg16_voc2007.yaml --bs 1 --nw 4 --iter_size 4
  ```
  
*Note: The current implementation only supports single-gpu training.*
