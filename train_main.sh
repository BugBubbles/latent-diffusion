export CUDA_VISIBLE_DEVICES=0,1,2,3
pyd main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml -t --gpus=0,1,2,3, --strategy=ddp_find_unused_parameters_true