# 9992project
# Attribute Group Editing for Reliable Few-shot Image Generation applied in the domain of emotion generation

## Description   
Modified implementation of AGE for few-shot image generation. Code is modified from [AGE](https://github.com/UniBester/AGE.git).

### modification
- removed the seed fixing during inference. Now one seed is used to generate one image. All seeds are preset.
- added guassian noise in the datapipe line
- added orthogonal loss for global dictionary A

## Getting Started
### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation

- Clone this repo  

### Pretrained pSp
Follow the [instructions](https://github.com/eladrich/pixel2style2pixel.git) to train a pSp model firsly. Or you can also directly download the [pSp pre-trained models](https://drive.google.com/drive/folders/1gTSghHGuwoj9gKsLc2bcUNF6ioFBpRWB?usp=sharing).

## Training
### Preparing your Data
- organize the file structure as follows:
  ```
  └── data_root
      ├── train                      
      |   ├── cate-id_sample-id.jpg                # train-img
      |   └── ...                                  # ...
      └── valid                      
          ├── cate-id_sample-id.jpg                # valid-img
          └── ...                                  # ...
  ```
The format of the file should be [label_id]_[any_name].jpg

## Repository structure
| Path | Description <img width=200>
| :--- | :---
| AGE | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects and augmentations
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in our experiments
| &boxvr; models | Folder containting all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing our pSp encoder architecture implementation and ArcFace encoder implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxur;&nbsp; age.py | Implementation of AGE
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; tools | Folder with running scripts for training and inference
| &boxvr;&nbsp; optimizer | Folder with Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxur;&nbsp; utils | Folder with various utility functions
| <img width=300> | <img>

- Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
- Refer to `configs/transforms_config.py` for the transforms defined for each dataset. 
- Finally, refer to `configs/data_configs.py` for the data paths for the train and valid sets
  as well as the transforms.
- To experiment with your own dataset, you can simply make the necessary adjustments in 
    1. `data_configs.py` to define your data paths.
    2. `transforms_configs.py` to define your own data transforms.


#### Get Class Embedding
To train AGE, the class embedding of each category in training set should be get first by using `tools/get_class_embedding.py`.
```
cd AGE; python tools/get_class_embedding.py \
--class_embedding_path=/path/to/save/classs/embeddings \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoint \
--train_data_path=/path/to/training/data \
--test_batch_size=4 \
--test_workers=4
```

### Training AGE
The main training script can be found in `tools/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

```
#set GPUs to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3

#begin training.
cd AGE; python -m torch.distributed.launch \
--nproc_per_node=4 \
tools/train.py \
--dataset_type=af_encode \
--exp_dir=/path/to/experiment/output \
--workers=8 \
--batch_size=8 \
--valid_batch_size=8 \
--valid_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--start_from_latent_avg \
--l2_lambda=1 \
--sparse_lambda=0.005 \
--orthogonal_lambda=0.0005 \
--A_length=100 \
--psp_checkpoint_path=/path/to/pretrained/pSp/checkpoint \
--class_embedding_path=/path/to/class/embeddings 
```

## Testing
### Inference
Having trained your model or using [pre-trained models](https://drive.google.com/drive/folders/17BZcbacTRSCPuapcLtVKQy9ZtTUzHfY_?usp=sharing) provided, you can use `tools/inference.py` to apply the model on a set of images.   
For example, 
```
python tools/inference.py \
--output_path=/path/to/output \
--checkpoint_path=/path/to/checkpoint \
--test_data_path=/path/to/test/input \
--train_data_path=/path/to/training/data \
--class_embedding_path=/path/to/classs/embeddings \
--n_distribution_path=/path/to/save/n/distribution \
--test_batch_size=4 \
--test_workers=4 \
--n_images=5 \
--alpha=1 \
--beta=0.005
```

### For emotion generation
specify different train_data_path and n_distribution_path for different emotion labels
```
python tools/inference.py \
--output_path=/path/to/output \
--checkpoint_path=/path/to/checkpoint \
--test_data_path=/path/to/test/input \
--train_data_path=/path/to/training/data/emotion02 \
--class_embedding_path=/path/to/classs/embeddings \
--n_distribution_path=/path/to/save/n/distribution/emotion02 \
--test_batch_size=4 \
--test_workers=4 \
--n_images=5 \
--alpha=1 \
--beta=0.005
```

## Citation
```
@inproceedings{ding2022attribute,
  title={Attribute Group Editing for Reliable Few-shot Image Generation},
  author={Ding, Guanqi and Han, Xinzhe and Wang, Shuhui and Wu, Shuzhe and Jin, Xin and Tu, Dandan and Huang, Qingming},
  booktitle=CVPR,
  year={2022},
}
```
