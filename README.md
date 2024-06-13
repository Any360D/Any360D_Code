### Environment
Our code is based on UniFuse (https://github.com/alibaba/UniFuse-Unidirectional-Fusion) and Depth Anything (https://github.com/LiheYoung/Depth-Anything).

Our code is based on Pytorch. The Pytorch environment we use: pytorch 2.0.0, torchvision 0.15.0, cuda 11.7

The python environment we use: Python 3.9.18

After installing the packages following UniFuse and Depth Anything, we can begin our experiments!

### Run the demo

1. Run a single 360 image:

```
python run_single_image.py --pred-only --img-path ./samples/360_image.jpg
```

2. Run a single 360 video:

```
python run_video.py --video-path ./samples/360_video.mp4
```

### Reproduce the results of Table. 1 (Representation)

```
python dam_representation.py --config ./configs/benchmark/representation_s.yaml
```

### Reproduce the results of Table. 2 (Mobius transformation)

```
python dam_mobius.py --config ./configs/benchmark/representation_s.yaml
```

Note that we put the checkpoint of Depth Anything with ViT-S backbone in the ./checkpoints. Testing other backbones requires to download manually from the Depth Anything Github.

### Reproduce the training process

We provide the corresponding training code and configs. However, before training, you need to prepare the dataset firstly.
1. Matterport3D dataset (You can follow UniFuse)
2. ZInd dataset (https://github.com/zillow/zind)
3. Download our Diverse360 dataset
Then, you can begin to reproduce our training process with three stages:
1. Train our teacher model;
2. Pseudo labeling;
3. Semi-supervision with both labeled and unlabed datasets.

We provide the checkpoints in the folder ./tmp/ of our two student models with different metric heads.