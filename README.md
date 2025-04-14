# PanOSIS - Panoramic One-Shot Image Segmentation
![image](https://github.com/user-attachments/assets/9382d901-9d57-48c7-8c80-d7d1aa168cb3)

**PanOSIS** is a one-shot image segmentation method for panoramic street-view images. 

![image](https://github.com/user-attachments/assets/06c59ac2-e07f-4a1d-929f-7f33303dc9fb)

## Table of contents
- [Datasets](#datasets)
- [Generation](#generation)
- [Segmentation](#segmentation)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)
- [Authors](#authors)

## Datasets
**CityScapes [(Cordts et al., 2016)](https://arxiv.org/abs/1604.01685)**
- 30 classes, 8 categories
- Flat street view images taken in 50 cities with varying weather conditions, infrastructure with their respective semantic segmentation masks

**CVRG-Pano [(Orhan, Bastanlar, 2021)](https://link.springer.com/article/10.1007/s11760-021-02003-3)**
- 20 classes, 7 categories
- Built based on Cityscapes 
- Panoramic street view images taken outdoors with their respective semantic segmentation masks

**One-Shot Streets [(Ference, Jung, 2025)](https://www.kaggle.com/datasets/jillference/one-shot-streets)**
- 20 classes, 7 categories
- Built based on CVRG-Pano 
- Panoramic street view images with their respective semantic segmentation masks generated from a single training pair from CVRG-Pano

## Generation 
[1] One-shot generation of an annotated panoramic street-view dataset
   
For one-shot generation of an annotated dataset, we used One-Shot Synthesis of Images and Segmentation Masks [(Shusko et al., 2022)](https://arxiv.org/abs/2209.07547)
- Generating new images and their segmentation masks from a single training pair
- Uses generative adversarial networks (GANs)
- Uses OSMIS model where the generator produces images and masks and the discriminator uses a masked content attention (MCA) module to assess alignment between images and masks

## Segmentation 
[2] Image segmentation of flat, panoramic, and synthetic panoramic data
   
### U-Net with Standard Convolutions ###
For image segmentation, we used UNet-stdconv that uses U-Net from UNet for Semantic Segmentation of Urban Scenes [(Heidari, 2023)](https://github.com/deepmancer/unet-semantic-segmentation)
- Uses convolutional neural networks (CNNs)

### U-Net with Equirectangular Convolutions ###
For panoramic image segmentation, we used UNet-equiconv that uses U-Net from UNet for Semantic Segmentation of Urban Scenes [(Heidari, 2023)](https://github.com/deepmancer/unet-semantic-segmentation) and a Pytorch implementation of EquiConv by [palver7](https://github.com/palver7/EquiConvPytorch)
- Uses convolutional neural networks (CNNs)
- Replaced standard convolution layers with equirectangular convolutions
- Move rectangular convolution kernels on the sphere representations of panoramic images

## Results
![image](https://github.com/user-attachments/assets/22fcae97-bff2-48d3-adf8-f8ad1baeb439)

![image](https://github.com/user-attachments/assets/9a45734c-4131-449c-8f3d-bd96f89b6a08)

![image](https://github.com/user-attachments/assets/da4c74f3-59f2-4208-9d6b-f5e96de17808)

## Deployment

1) Generation
- Follow steps from [Shusko's one-shot implementation](https://github.com/boschresearch/one-shot-synthesis?tab=readme-ov-file)
- Note on compatible dependencies: kornia 0.6.8, torch 2.4.1+cu121, conda 12.1
- Use converter.py to convert the generated RGB masks into class label masks so they can be better used in the image segmentation

2) Segmentation
- Run the generated dataset on UNet from unet_stdconv or unet_equiconv

## License

MIT License

## Acknowledgements
For this project, we built upon [Shusko](https://github.com/boschresearch/one-shot-synthesis?tab=readme-ov-file)’s model with our inclusion of a one-shot synthesis of an annotated panoramic street-view images dataset. We also built upon [Heidari](https://github.com/deepmancer/unet-semantic-segmentation)’s U-Net with our implementation of UNet-stdconv. Furthermore, we built upon [palver7](https://github.com/palver7/EquiConvPytorch)’s Pytorch implementation of EquiConv for our implementation of UNet-equiconv.

## Authors
- Jung, Seolin ([@seolinjung](https://github.com/seolinjung))
- Ference, Jill ([@JillFerence](https://github.com/JillFerence))
