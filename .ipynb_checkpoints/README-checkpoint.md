
# pix2pix in maps generation


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ChongjianGE/maps_generation.git
cd maps_generation
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.



### pix2pix train/test
- Prepare the maps dataset (e.g.[maps](https://drive.google.com/file/d/15fziaX7zUI1iHDBfGWflB5RVibIhcXGz/view)):
```bash
tar -xvf maps.tar
```

- Train a model:
```
python3 train_pix2pix.py --name map_translation --gpu_ids 0 --dataset_mode map
```
- To see more intermediate results, check out  
```
/apdcephfs/share_1290939/chongjiange/train_output/ckpt/map_translation (for checkpoints)
/apdcephfs/share_1290939/chongjiange/train_output/runs/map_translation (for tensorboard)
```

- Test the model :
```
python3 test_model_general.py --name map_translation --num_test 30000 --dataset_mode map --gpu_ids 0
```

### Sth to Notice

- dataset preparation
    - you can modify the `data/map_dataset.py` file for your own configuration

- model
    - you can modify the `models/pix2pix_model.py` file for model design
    - you can modify the `models/networks.py` file for subnetwork design
    
- learning configuration
    - you can modify the files in `option/`  for modifying learning configurations


