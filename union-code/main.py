import os
import time

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from utils.image import imread
from utils.video import read_frames_from_dir
from config import *
from datasets.cropset import CropSet
from datasets.frameset import FrameSet
from datasets.interpolation_frameset import TemporalInterpolationFrameSet
from diff.conditional_diffusion import ConditionalDiffusion
from diff.diffusion import Diffusion
from union.convchain import ConvChain


def summarize_model_params(model):
    """
    Display total parameter count (in millions) of the given model.
    """
    param_count = sum(param.numel() for param in model.parameters())
    print(f"Model has {param_count / 1e6:.2f} million parameters")


def train_single_image(cfg):
    """
    Train a diffusion model on a single image as specified in 'cfg'.
    """
    max_steps = 50_000
    image_tensor = imread(f'./images/{cfg.image_name}')

    # Create and display model stats
    unet_model = ConvChain(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    summarize_model_params(unet_model)

    # Prepare data loaders
    crop_dim = int(min(image_tensor[0].shape[-2:]) * 0.95)
    dataset = CropSet(image=image_tensor, crop_size=crop_dim, use_flip=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    # Instantiate model and diffusion wrapper
    unet_model = ConvChain(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion_module = Diffusion(
        unet_model,
        training_target='x0',
        timesteps=cfg.diffusion_timesteps,
        auto_sample=True,
        sample_size=image_tensor[0].shape[-2:]
    )

    # Set up callbacks
    callback_list = [
        pl.callbacks.ModelSummary(max_depth=-1),
        pl.callbacks.ModelCheckpoint(
            filename='single-level-{step}',
            save_last=True,
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        )
    ]

    # Set up logger and trainer
    logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=cfg.image_name,
        version=cfg.run_name
    )
    trainer = pl.Trainer(
        max_steps=max_steps,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        log_every_n_steps=10,
        callbacks=callback_list
    )

    # Begin training
    trainer.fit(diffusion_module, loader)


def train_video_frame_predictor(cfg):
    """
    Train a DDPM-based frame predictor for a video stored in './images/video/<cfg.image_name>'.
    """
    max_steps = 200_000
    frames_seq = read_frames_from_dir(f'./images/video/{cfg.image_name}')

    # Create dataset
    h_size = int(frames_seq[0].shape[-2] * 0.95)
    w_size = int(frames_seq[0].shape[-1] * 0.95)
    ds_predict = FrameSet(frames=frames_seq, crop_size=(h_size, w_size))
    loader = DataLoader(ds_predict, batch_size=1, num_workers=4, shuffle=True)

    # Build model and show param count
    predictor_model = ConvChain(in_channels=6, filters_per_layer=cfg.network_filters, depth=cfg.network_depth, frame_conditioned=True)
    summarize_model_params(predictor_model)

    # Set up conditional diffusion
    cond_diff = ConditionalDiffusion(
        predictor_model,
        training_target='noise',
        noise_schedule='cosine',
        timesteps=cfg.diffusion_timesteps
    )

    # Trainer callbacks
    cbs = [
        pl.callbacks.ModelSummary(max_depth=-1),
        pl.callbacks.ModelCheckpoint(
            filename='single-level-{step}',
            save_last=True,
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        )
    ]

    # Logger + Trainer
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_predictor')
    trainer = pl.Trainer(
        max_steps=max_steps,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        log_every_n_steps=10,
        callbacks=cbs
    )

    # Train
    trainer.fit(cond_diff, loader)


def train_video_frame_projector(cfg):
    """
    Train a frame projector model on video frames from './images/video/<cfg.image_name>'.
    """
    max_steps = 100_000
    video_frames = read_frames_from_dir(f'./images/video/{cfg.image_name}')

    # Crop size determination
    crop_dim = int(min(video_frames[0].shape[-2:]) * 0.95)

    # Create and print model parameters
    projector_model = ConvChain(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    summarize_model_params(projector_model)

    # Prepare dataset + loader
    ds_project = CropSet(image=video_frames, crop_size=crop_dim, use_flip=False)
    loader = DataLoader(ds_project, batch_size=1, num_workers=4, shuffle=True)

    # Diffusion
    projector_model = ConvChain(in_channels=3, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diff_module = Diffusion(
        projector_model,
        training_target='noise',
        noise_schedule='cosine',
        timesteps=cfg.diffusion_timesteps
    )

    # Callbacks and trainer
    callbacks_ = [
        pl.callbacks.ModelSummary(max_depth=-1),
        pl.callbacks.ModelCheckpoint(
            filename='single-level-{step}',
            save_last=True,
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        )
    ]
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_projector')
    trainer = pl.Trainer(
        max_steps=max_steps,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        log_every_n_steps=10,
        callbacks=callbacks_
    )

    # Train
    trainer.fit(diff_module, loader)


def train_video_frame_interpolator(cfg):
    """
    Train a DDPM-based interpolator model for temporally filling in between frames in a video.
    """
    max_steps = 50_000
    frame_list = read_frames_from_dir(f'./images/video/{cfg.image_name}')
    crop_dim = int(min(frame_list[0].shape[-2:]) * 0.95)

    # Create model
    interpolator_net = ConvChain(in_channels=9, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    summarize_model_params(interpolator_net)

    # Prepare dataset and diffusion
    ds_inter = TemporalInterpolationFrameSet(frames=frame_list, crop_size=crop_dim)
    loader = DataLoader(ds_inter, batch_size=1, num_workers=4, shuffle=True)

    interpolator_net = ConvChain(in_channels=9, filters_per_layer=cfg.network_filters, depth=cfg.network_depth)
    diffusion_module = ConditionalDiffusion(
        interpolator_net,
        training_target='x0',
        timesteps=cfg.diffusion_timesteps
    )

    # Set up callbacks and trainer
    cbacks = [
        pl.callbacks.ModelSummary(max_depth=-1),
        pl.callbacks.ModelCheckpoint(
            filename='single-level-{step}',
            save_last=True,
            save_top_k=3,
            monitor='train_loss',
            mode='min'
        )
    ]
    logger = pl.loggers.TensorBoardLogger("lightning_logs/", name=cfg.image_name, version=cfg.run_name + '_interpolator')
    trainer = pl.Trainer(
        max_steps=max_steps,
        gpus=1,
        auto_select_gpus=True,
        logger=logger,
        log_every_n_steps=10,
        callbacks=cbacks
    )

    # Train
    trainer.fit(diffusion_module, loader)


def main():
    start_time = time.time()
    cfg = BALLOONS_IMAGE_CONFIG
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    # Choose the task
    if cfg.task == 'video':
        train_video_frame_predictor(cfg)
        train_video_frame_projector(cfg)
    elif cfg.task == 'video_interp':
        train_video_frame_interpolator(cfg)
        train_video_frame_projector(cfg)
    elif cfg.task == 'image':
        train_single_image(cfg)
    else:
        raise ValueError(f"Unrecognized task: {cfg.task}")

    # Print total execution time
    end_time = time.time()
    duration_seconds = end_time - start_time
    print(f"Total execution time: {duration_seconds / 60:.2f} minutes.")

    # Inline callback class to track and plot training loss
    class LossPlotterCallback(Callback):
        def __init__(self):
            super().__init__()
            self.loss_history = []

        def on_train_epoch_end(self, trainer, pl_module):
            # 'train_loss' should be logged inside the training loop
            current_loss = trainer.callback_metrics.get('train_loss')
            if current_loss is not None:
                self.loss_history.append(current_loss.item())

        def on_train_end(self, trainer, pl_module):
            # Plot the collected losses
            plt.figure(figsize=(8, 6))
            plt.plot(self.loss_history, marker='o', label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss over Epochs")
            plt.legend()
            plt.grid(True)
            plt.savefig("training_loss.png")
            plt.show()
            print("Saved loss plot to training_loss.png")


if __name__ == '__main__':
    main()
