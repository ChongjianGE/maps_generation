
# pix2pix in 2.5Dtryon


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Mosasaur5526/2.5DVirtualTryon.git
cd 2.5DVirtualTryon
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.



### pix2pix train/test
- Prepare the 2.5dtryon dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
python ft_local/process.py
```

- Train a model:
```
bash start.sh (for 8 gpus)
python3 train_pix2pix.py --name 3d_try_new_rightdata_8gpu --gpu_ids 0 --dataset_mode tryon --gpu_ids 0,1,2,3,4,5,6,7
```
- To see more intermediate results, check out  
```
/apdcephfs/share_1290939/chongjiange/train_output/3dtryon/ckpt/ (for checkpoints)
/apdcephfs/share_1290939/chongjiange/train_output/3dtryon/runs/ (for tensorboard)
```

- Test the model :
```
WIP
```

### Sth to Notice

- dataset preparation
    - you can modify the `data/tryon_dataset.py` file for your own configuration

- model
    - you can modify the `models/pix2pix_model.py` file for model design
    - you can modify the `models/networks.py` file for subnetwork design

