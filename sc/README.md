# [SimpleClick: Interactive Image Segmentation with Simple Vision Transformers (Jittor)](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SimpleClick_Interactive_Image_Segmentation_with_Simple_Vision_Transformers_ICCV_2023_paper.html)

**University of North Carolina at Chapel Hill**

[Qin Liu](https://sites.google.com/cs.unc.edu/qinliu/home), [Zhenlin Xu](https://wildphoton.github.io/), [Gedas Bertasius](https://www.gedasbertasius.com/), [Marc Niethammer](https://biag.cs.unc.edu/)

ICCV 2023

<p align="center">
    <a href="https://paperswithcode.com/sota/interactive-segmentation-on-sbd?p=simpleclick-interactive-image-segmentation">
        <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/simpleclick-interactive-image-segmentation/interactive-segmentation-on-sbd"/>
    </a>
</p>

<p align="center">
  <img src="./assets/simpleclick_framework.png" alt="drawing", width="650"/>
</p>


## Environment
Training and evaluation environment: Python3.8.8, PyTorch 1.11.0, Ubuntu 20.4, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```
You can build a container with the configured environment using our [Dockerfiles](https://github.com/uncbiag/SimpleClick/tree/v1.0/docker). 
Our Dockerfiles only support CUDA 11.0/11.4/11.6. If you use different CUDA drivers, you need to modify the base image in the Dockerfile (This is annoying that you need a matched image in Dockerfile for your CUDA driver, otherwise the gpu doesn't work in the container. Any better solutions?).
You also need to configue the paths to the datasets in [config.yml](https://github.com/uncbiag/SimpleClick/blob/v1.0/config.yml) before training or testing.

## Demo
<p align="center">
  <img src="./assets/demo_sheep.gif" alt="drawing", width="500"/>
</p>

An example script to run the demo. 
```
python3 demo.py --checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth --gpu 0
```
Some test images can be found [here](https://github.com/uncbiag/SimpleClick/tree/v1.0/assets/test_imgs).

## Evaluation
Before evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/uncbiag/SimpleClick/blob/v1.0/config.yml).

Use the following code to evaluate the huge model.
```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=./weights/simpleclick_models/cocolvis_vit_huge.pth \
--eval-mode=cvpr \
--datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD,COCO_MVal,ssTEM,BraTS,OAIZIB
```

## Training
Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)) and configure the dowloaded path in [config.yml](https://github.com/uncbiag/SimpleClick/blob/main/config.yml).

Use the following code to train a huge model on C+L: 
```
python train.py models/iter_mask/plainvit_huge448_cocolvis_itermask.py \
--batch-size=32 \
--ngpus=4
```

## Download 
SimpleClick models: [Google Drive](https://drive.google.com/drive/folders/1qpK0gtAPkVMF7VC42UA9XF4xMWr5KJmL?usp=sharing)

BraTS dataset (369 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing) 

OAI-ZIB dataset (150 cases): [Google Drive](https://drive.google.com/drive/folders/1B6y1nNBnWU09EhxvjaTdp1XGjc1T6wUk?usp=sharing)

Other datasets: [RITM Github](https://github.com/saic-vul/ritm_interactive_segmentation)

## Notes
[03/11/2023] Add an xTiny model.

[10/25/2022] Add docker files.

[10/02/2022] Release the main models. This repository is still under active development.

## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Citation
```bibtex
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Qin and Xu, Zhenlin and Bertasius, Gedas and Niethammer, Marc},
    title     = {SimpleClick: Interactive Image Segmentation with Simple Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22290-22300}
}
```
## Acknowledgement
Our project is developed based on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation). Thanks for the nice demo GUI :)
