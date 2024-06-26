# Union - Single Image/Video Generation

## requirements

pytorch_lightning==1.5.10<br>
Pytorch 1.11.0<br>
torchvision==0.12.0
```
pip install -r requirements.txt
```
## train

For image training:```python main.py --task='image' --image_name='image0.png' --run_name='image_0'image```<br>
For video training: ```python main.py --task='video' --image_name='balloons' --run_name='balloons_video_model_0'```

## datasets
The image dataset has been uploaded. The video dataset comes from HP-VAE-GAN, SinGAN-GIF and VGPNN.
