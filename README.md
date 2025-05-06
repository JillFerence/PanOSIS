# PanOSIS - Panoramic One-Shot Image Segmentation
![image](https://github.com/user-attachments/assets/393241a2-711a-479c-acc1-fcdd982a3d5d)


**PanOSIS** is a one-shot image segmentation method for panoramic street-view images. 

- Task: One-shot image segmentation for panoramic street view images
- Problem: Despite the potential for product development and urban environment analysis, panorama suffers from limited annotated datasets for segmentation purposes and lack of research
- Solution:
  - One-shot synthesis: Generating annotated panoramic dataset from a single image-mask pair
   → Addressing dataset limitations
  - Equirectangular convolutions: Modifying conventional models (i.e. U-Net) to better conform to panoramic distortions
   → Improving segmentation accuracy
- Contribution: Demonstrating the potential of one-shot learning for panoramic image segmentation

![image](https://github.com/user-attachments/assets/e4400d18-dfbd-49a1-a7e2-e521d52a4009)


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
### Image Generation

![image](https://github.com/user-attachments/assets/90547b6e-0773-404c-9ce5-67a68fac114e)

### Image Segmentation

![image](https://github.com/user-attachments/assets/5b857308-7140-4935-a996-91617fd86eed)

![image](https://github.com/user-attachments/assets/197d0b1f-67d5-4ced-8ac8-b343636a3dc8)

![image](https://github.com/user-attachments/assets/3f2889bc-63d7-4753-b9dd-534d3657e26a)

![image](https://github.com/user-attachments/assets/30282807-ca28-4d14-b634-fe3779740b38)

![image](https://github.com/user-attachments/assets/3c0dc905-0c33-4fcc-a6ec-4e6c210f8973)

## Deployment

1) Generation
- Follow steps from [Shusko's one-shot implementation](https://github.com/boschresearch/one-shot-synthesis?tab=readme-ov-file)
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
