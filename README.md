# Union - Single Image/Video Generation

## Requirements

pytorch_lightning==1.5.10<br>
Pytorch 1.11.0<br>
torchvision==0.12.0

## Train

For image training:```python main.py --task='image' --image_name='image0.png' --run_name='image_0'```<br>
For video training: ```python main.py --task='video' --image_name='balloons' --run_name='balloons_video_model_0'```

## Datasets
The image dataset has been uploaded. The video dataset comes from HP-VAE-GAN, SinGAN-GIF and VGPNN.

## Implementation Notes  
This project extends the following open-source works:  
- **Sinddm** [[Code]](https://github.com/fallenshock/SinDDM.git): Adopted for the backbone diffusion model architecture.  
- **Sindiffusion** [[Code]](https://github.com/WeilunWang/SinDiffusion.git): Utilized for training pipeline design.  
- **NNFdiversity** [[Code]](https://github.com/nivha/nnf_diversity.git): Adapted the NNF metric implementation.  
