# Long-Exposure Fusion

## Unfinished

This repository is unfinished and will be ready to use soon.
Don't try running the project just yet.

## Introduction

**Long-Exposure Fusion** is a full Python pipeline that generates long-exposure style images from videos or burst photo sequences. Images are generated using our own variant of [Exposure Fusion](https://ieeexplore.ieee.org/document/4392748) that we call Hybrid Weight-Map Fusion.

This repository provides:
- An automated pipeline for decoding, aligning, interpolating, defining segmentation masks, and fusing images.
- A simple GUI for segmenting images.
- Command-line tools for processing your own videos or image bursts.

## Installation

### Disclaimer

This repository is brand new and you may encounter problems before you get everything running.
Don't hesitate to open a Github issue if you do.

The steps below describe how to install this project on a Linux system with python 3.11.13 and cuda and were tested using WSL.
Try using a different environment at your own risk.

### Source

First, clone this repository using:
```bash
git clone https://github.com/ThomasMPont/long-exposure-fusion.git
cd long-exposure-fusion
```

Then ensure you have python 3.11.13 installed:
```bash
python3.11 --version
```
If you don't, you'll want to run something like:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

Unless you know what you're doing. You'll want to create a virtual environment using:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

Once in your venv, run:
```bash
pip install -r requirements.txt
```

### Submodules

This repository depends on forks of [SAM2](https://github.com/ThomasMPont/sam2-sequential) and [Practical-RIFE](https://github.com/ThomasMPont/Practical-RIFE).
Install them using:
```bash
git submodule update --init
```

#### Practical-RIFE

Download [this model](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing) and paste `*.py` and `flownet.pkl` into `external/Practical-RIFE/train_log/`.  
You may also use any other of the [pretrained Practical-RIFE models](https://github.com/ThomasMPont/Practical-RIFE#trained-model).

#### SAM2 

Download [this model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and paste the `.pt`file into `external/sam2-sequential/checkpoints/`.  
You may also use any other of the [pretrained SAM2 models](https://github.com/ThomasMPont/sam2-sequential?tab=readme-ov-file#download-checkpoints).
If you choose to do so, you'll have to edit the model path in `src/pipeline/segment_picker.py`.
