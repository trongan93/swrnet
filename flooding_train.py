import sys, os
from pathlib import Path
import torch
from ml4floods.models.config_setup import get_default_config
import pkg_resources
from ml4floods.models.dataset_setup import get_dataset
import importlib
import matplotlib.pyplot as plt
from models import flooding_model
flooding_model = importlib.reload(flooding_model)
import copy
from models.flooding_model import WorldFloodsModel, DistilledTrainingModel, WorldFloodsModel2, WorldFloodsModel1
importlib.reload(flooding_model)
from ptflops import get_model_complexity_info
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
import numpy as np
from ml4floods.models.utils import metrics
from ml4floods.models.model_setup import get_model_inference_function
import pandas as pd


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Load flooding configuration file from local device or gcs
config_fp = pkg_resources.resource_filename("ml4floods","models/configurations/worldfloods_template.json")

config = get_default_config(config_fp)
config

# Step 2: Setup Dataloader

config.experiment_name = 'training_flooding_bgri'
config.data_params.channel_configuration = 'bgri'
config.model_params.hyperparameters.channel_configuration = 'bgri'
config.model_params.hyperparameters.num_channels = 4
config.data_params.bucket_id = ""
config.model_params


config.data_params.batch_size = 128 # control this depending on the space on your GPU!
config.data_params.loader_type = 'local'
config.data_params.path_to_splits = "/mnt/d/Flooding/worldfloods_v1_0" # local folder to download the data
config.data_params.train_test_split_file = "/mnt/d/Flooding/train_test_split_local.json"

config.data_params["download"] = {"train": True, "val": True, "test": True} # download only test data
# config.data_params.train_test_split_file = "2_PROD/2_Mart/worldfloods_v1_0/train_test_split.json" # use this to train with all the data
config.data_params.num_workers = 24

# If files are not in config.data_params.path_to_splits this will trigger the download of the products.
dataset = get_dataset(config.data_params)

# Verfify data loader
train_dl = dataset.train_dataloader()
train_dl_iter = iter(train_dl)
print(len(train_dl_iter))
batch_train = next(train_dl_iter)

val_dl = dataset.val_dataloader()
val_dl_iter = iter(val_dl)
print(len(val_dl_iter))
batch_val = next(val_dl_iter)

test_dl = dataset.test_dataloader()
test_dl_iter = iter(test_dl)
print(len(test_dl_iter))
batch_test = next(test_dl_iter)

#Plot batch by using ml4flood model

#Step 3: Setup Model

config.model_params

config.model_params.model_folder = "train_models" 
os.makedirs("train_models", exist_ok=True)
config.model_params.test = False
config.model_params.train = True
config.model_params.hyperparameters.model_type = "unet_simple" # Currently implemented: simplecnn, unet, linear, unet_simple
config.model_params.hyperparameters.metric_monitor = 'val_iou_loss' #IoU Loss

simple_model_params = copy.deepcopy(config.model_params)
# simple_model_params['hyperparameters']['model_type']="unet_simple"

# model = DistilledTrainingModel(config.model_params, simple_model_params)
model = WorldFloodsModel2(config.model_params) # Focal loss and IoU loss
# model = WorldFloodsModel1(config.model_params) # Focal loss and Dice loss
net = model.network
net

# Compuatation complexity of network

macs, params = get_model_complexity_info(net, (config.model_params.hyperparameters.num_channels, config.model_params.hyperparameters.max_tile_size, config.model_params.hyperparameters.max_tile_size), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

setup_weights_and_biases = False #TRue mawjc didnh
if setup_weights_and_biases:
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    # UNCOMMENT ON FIRST RUN TO LOGIN TO Weights and Biases (only needs to be done once)
    wandb.login()
    run = wandb.init()

    # Specifies who is logging the experiment to wandb
    config['wandb_entity'] = 'ml4floods'
    # Specifies which wandb project to log to, multiple runs can exist in the same project
    config['wandb_project'] = 'worldfloods-trongan-test'

    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project, 
        entity=config.wandb_entity
    )
else:
    wandb_logger = None
    
experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{experiment_path}/checkpoint",
    save_top_k=True,
    verbose=True,
    monitor='val_iou_loss',
    mode='min',
#     prefix=''
)

early_stop_callback = EarlyStopping(
    monitor='val_iou_loss',
    patience=10,
    strict=False,
    verbose=False,
    mode='min'
)
# monitor='val_iou_loss'
# monitor='val_dice_loss'


callbacks = [checkpoint_callback, early_stop_callback]

print(f"The trained model will be stored in {config.model_params.model_folder}/{config.experiment_name}")

config.gpus = 2 # which gpu to use 
# config.gpus = None # to not use GPU
config.model_params.hyperparameters.max_epochs = 5 # train for maximum 4 epochs

trainer = Trainer(
    fast_dev_run=False,
    logger=wandb_logger,
    callbacks=callbacks,
    default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
    accumulate_grad_batches=1,
    gradient_clip_val=0.0,
    auto_lr_find=False,
    benchmark=False,
    max_epochs=config.model_params.hyperparameters.max_epochs,
    check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
    strategy='dp',
    accelerator='gpu',
    devices=config.gpus
    # resume_from_checkpoint='~/Projects/github/satellite-knowledge-distillation/train_models/training_flooding_bgri/checkpoint/epoch=0-step=24269.ckpt'
)

trainer.fit(model, dataset)

# Run inference on the images shown before

logits = model(batch_train["image"].to(model.device))
print(f"Shape of logits: {logits.shape}")
probs = torch.softmax(logits, dim=1)
print(f"Shape of probs: {probs.shape}")
prediction = torch.argmax(probs, dim=1).long().cpu()
print(f"Shape of prediction: {prediction.shape}")

config.model_params.max_tile_size = config.model_params.hyperparameters.max_tile_size
config

# model.to("cuda")
inference_function = get_model_inference_function(model, config, apply_normalization=False, activation="softmax")

# dataset2 = get_dataset(config.data_params)
dl = dataset.val_dataloader() # pytorch Dataloader
print(str(dl.batch_size))

# Otherwise fails when reading test dataset from remote bucket
# torch.set_num_threads(1)

thresholds_water = [0,1e-3,1e-2]+np.arange(0.5,.96,.05).tolist() + [.99,.995,.999]

mets = metrics.compute_metrics(
    dl,
    inference_function, 
    thresholds_water=thresholds_water, 
    plot=False, convert_targets=False)

label_names = ["land", "water", "cloud"]
# metrics.plot_metrics(mets, label_names)
metrics_df = pd.DataFrame(mets)
output_file = "metrics_results.csv"
metrics_df.to_csv(output_file, index=False)
print(f"Metrics have been saved to {output_file}")


if hasattr(dl.dataset, "image_files"):
    cems_code = [os.path.basename(f).split("_")[0] for f in dl.dataset.image_files]
else:
    cems_code = [os.path.basename(f.file_name).split("_")[0] for f in dl.dataset.list_of_windows]

iou_per_code = pd.DataFrame(metrics.group_confusion(mets["confusions"],cems_code, metrics.calculate_iou,
                                                    label_names=[f"IoU_{l}"for l in ["land", "water", "cloud"]]))

recall_per_code = pd.DataFrame(metrics.group_confusion(mets["confusions"],cems_code, metrics.calculate_recall,
                                                       label_names=[f"Recall_{l}"for l in ["land", "water", "cloud"]]))

join_data_per_code = pd.merge(recall_per_code,iou_per_code,on="code")
join_data_per_code = join_data_per_code.set_index("code")
join_data_per_code = join_data_per_code*100
print(f"Mean values across flood events: {join_data_per_code.mean(axis=0).to_dict()}")
join_data_per_code
output_file = "flood_event_metrics.csv"
join_data_per_code.to_csv(output_file)
print(f"Metrics per flood event have been saved to {output_file}")



torch.save(model.state_dict(),f"{experiment_path}/model_irirnir_worldflood_model_1_epoch_2_adaptive_gamma_alpha_0_001.pt")
# Save cofig file in experiment_path
config_file_path = f"{experiment_path}/config_irirnir_worldflood_model_1_epoch_2_adaptive_gamma_alpha_0_001.json"
import json
with open(config_file_path, 'w') as f:
    json.dump(config, f)

# Run inference on the images shown before

logits = model(batch_val["image"].to(model.device))
print(f"Shape of logits: {logits.shape}")
probs = torch.softmax(logits, dim=1)
print(f"Shape of probs: {probs.shape}")
prediction = torch.argmax(probs, dim=1).long().cpu()
print(f"Shape of prediction: {prediction.shape}")


# import matplotlib.pyplot as plt
# import importlib

n_image_start = 7
n_images = 14
count = int(n_images - n_image_start)

fig, axs = plt.subplots(4, count, figsize=(18, 14), tight_layout=True)

importlib.reload(flooding_model)

flooding_model.plot_batch(
    batch_val["image"][n_image_start:n_images],
    channel_configuration="bgri",
    axs=axs[0],
    max_clip_val=3500.
)
flooding_model.plot_batch(
    batch_val["image"][n_image_start:n_images],
    channel_configuration="bgri",
    bands_show=["B8", "B8", "B8"],
    axs=axs[1],
    max_clip_val=3500.
)
flooding_model.plot_batch_output_v1(
    batch_val["mask"][n_image_start:n_images, 0],
    axs=axs[2],
    show_axis=True
)
flooding_model.plot_batch_output_v1(
    prediction[n_image_start:n_images] + 1,
    axs=axs[3],
    show_axis=True
)

for ax in axs.ravel():
    ax.grid(False)

output_file = "batch_visualization.png"
plt.savefig(output_file, dpi=300)
print(f"Visualization has been saved to {output_file}")

plt.close(fig)
