# EDSDF
EDSDF: An Efficient Deep Supervised Distillation Framework for Medical Image Segmentation

## File Descriptions

- `make_json.py`: Used to generate JSON files containing dataset paths.
- `train_EDSDF.py`: Used to train the EDSDF model.
- `train_EDSDF_KD.py`: Used to train the student model using KAMSD.
- `eval.py`: Used to evaluate model performance.

## Architecture

### EDSDF
<p align="center">
<img src="pic/Architecture1.png" width=80% 
class="center">
</p>

### Component
<p align="center">
<img src="pic/Architecture2.png" width=81% 
class="center">
</p>

### KAMSD
<p align="center">
<img src="pic/Architecture3.png" width=80% 
class="center">
</p>

## Results
<p align="center">
<img src="pic/Results_1.png" width=100% 
class="center">
</p>

<p align="center">
<img src="pic/Results_2.png" width=100% 
class="center">
</p>
Qualitative comparison of EDSDF with other methods. (1) Input images. (2) Ground Truth. (3) U-net. (4) FCN. (5) AttU-Net. (6) Unet++. (7) DeepLabV3+. (8) CompNet. (9) CE-Net. (10) BiSeNetV2. (11) FAT-Net. (12) HiFormer. (13) CMUNeXt. (14) ConvUNeX. (15) EMCAD. (16) MADGNet. (17) Segformer. (18) Unext. (19) MedNeXt. (20) CFANet. (21) CASFNet. (22) EDSDF-S(w/o KAMSD). (23) EDSDF-S. (24) EDSDF-T. The green and red lines represent the ground truth and predicted boundaries, respectively.

## Feature Map Vis
<p align="center">
<img src="pic/Feature_map_vis.png" width=50% 
class="center">
</p>
