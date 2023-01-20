
LaneATT

This repository holds the source code for LaneATT, a novel state-of-the-art lane detection model proposed in the [paper](https://arxiv.org/abs/2010.12035) "_Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection_", 
by [Lucas Tabelini](https://github.com/lucastabelini), [Rodrigo Berriel](http://rodrigoberriel.com), [Thiago M. Paixão](https://sites.google.com/view/thiagopx), [Claudine Badue](http://www.inf.ufes.br/~claudine/)

Table of contents
1. [Prerequisites]
2. [Install]
3. [Steps to run the code]
4. [Description of my contribution]
5. [Code structure]

1. Prerequisites
- Python >= 3.5
- PyTorch == 1.6, tested on CUDA 10.2
- CUDA, to compile the NMS code
- Other dependencies described in `requirements.txt`

2. Install
```bash
conda create -n laneatt python=3.8 -y
conda activate laneatt
conda install pytorch==1.6 torchvision -c pytorch
pip install -r requirements.txt
cd lib/nms; python setup.py install; cd -
```

3. Steps to run the code
#### Dataset - Tusimple
Inside the code's root directory, run the following:

```bash
mkdir datasets # if it does not already exists
cd datasets
# train & validation data (~10 GB)
mkdir tusimple
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple
# test images (~10 GB)
mkdir tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test
# test annotations
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/
cd ..
```

To train LaneATT with the ResNet-34 backbone on TuSimple, run:
```
python main.py train --exp_name laneatt_r34_tusimple --cfg cfgs/laneatt_tusimple_resnet34.yml

```
After running this command, a directory `experiments` should be created (if it does not already exists). Another
directory `laneatt_r34_tusimple` will be inside it, containing data related to that experiment (e.g., model checkpoints, logs, evaluation results, etc)

Evaluate a model:
```
python main.py test --exp_name laneatt_r34_tusimple --save_predictions
```
Save the resulting frames:

```
python gen_video.py --pred ./predictions.pkl --cfg ./cfgs/laneatt_tusimple_resnet34.yml --view
```
4. Description of my contribution
1. gen_video.py - /LaneATT/gen_video.py
   - Generates frames from TuSimple model's predictions
2. Modifications in laneatt.py:
These are the starting lines of modifications implemented by me in the code:

line 30: Image normalization technique:
Output channel = 255 ∗ (Input channel − min)/(max − min)

line 64: 
self.conv2d = nn.LazyConv2d(1, kernel size = 3, padding = 1) 

line 328:
torch.nn.init.xavier_normal_(layer.weight, gain=0.001)

line 75:
Squeeze and excitation module


5. Code structure
- **cfgs:** Default configuration files
- **figures:** Images used in this repository
- **lib**
  - **datasets**
    - **culane.py:** CULane annotation loader
    - **lane_dataset.py:** Transforms raw annotations from a `LaneDatasetLoader` into a format usable by the model
    - **lane_dataset_loader.py:** Abstract class that each dataset loader implements
    - **llamas.py:** LLAMAS annotation loader
    - **nolabel_dataset.py:** Used on data with no annotation available (or quick qualitative testing)
    - **tusimple.py:** TuSimple annotation loader
   - **models:**
     - **laneatt.py:** LaneATT implementation
     - **matching.py:** Utility function for ground-truth and proposals matching
     - **resnet.py:** Implementation of ResNet
  - **nms:** NMS implementation
  - **config.py:** Configuration loader
  - **experiment.py:** Tracks and stores information about each experiment
  - **focal_loss.py:** Implementation of Focal Loss
  - **lane.py:** Lane representation
  - **runner.py:** Training and testing loops
- **utils**:
  - **culane_metric.py:** Unofficial implementation of the CULane metric. This implementation is faster than the oficial,
  however, it does not matches exactly the results of the official one (error in the order of 1e-4). Thus, it was used only during the model's development.
  For the results reported in the paper, the official one was used.
  - **gen_anchor_mask.py**: Computes the frequency of each anchor in a dataset to be used in the anchor filtering step
  - **gen_video.py:** Generates a video from a model's predictions
  - **llamas_metric.py**: Official implementation of the LLAMAS metric
  - **llamas_utils.py**: Utilities functions for the LLAMAS dataset
  - **speed.py:** Measure efficiency-related metrics of a model
  - **tusimple_metric.py**: Official implementation of the TuSimple metric
  - **viz_dataset.py**: Show images sampled from a dataset (post-augmentation)
- **main.py:** Runs the training or testing phase of an experiment

