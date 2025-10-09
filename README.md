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

Create a virtual environment using:
```bash
python3.11 -m venv venv
source venv/bin/activate
```
All of the following steps should be run inside this venv.

Install the basic modules using:
```bash
pip install -r requirements.txt
```

### Dependencies

This repository depends on forks of [LightGlue](https://github.com/cvg/LightGlue) [Practical-RIFE](https://github.com/ThomasMPont/Practical-RIFE), and [SAM2](https://github.com/ThomasMPont/sam2-sequential).
Install them using:
```bash
git submodule update --init
```

#### LightGlue

Install LightGlue from the `LightGlue` submodule:
```bash
cd external/LightGlue
pip install -e .
```

#### Practical-RIFE

Download [this model](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing) and 
`train_log/` into `external/Practical-RIFE/`.  
You may also use any other of the [pretrained Practical-RIFE models](https://github.com/ThomasMPont/Practical-RIFE#trained-model).

#### SAM2

SAM2 requires compiling a custom CUDA kernel with the nvcc compiler. If it isn't already available on your machine, please install [the CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.

Install SAM2 from the `sam2-sequential` submodule:
```bash
cd external/sam2-sequential
pip install -e .
```

Download [this model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and paste the `.pt`file into `external/sam2-sequential/checkpoints/`.  
You may also use any other of the [pretrained SAM2 models](https://github.com/ThomasMPont/sam2-sequential?tab=readme-ov-file#download-checkpoints).
If you choose to do so, you'll have to edit the model path in `src/pipeline/segment_picker.py`.

#### PyTorch

Install PyTorch and Torchvision using:
```bash
pip3 install torch torchvision
```

#### FFMPEG

To decode video inputs, you'll need to install FFMPEG:
```bash
sudo apt update
sudo apt install ffmpeg
```

## Demo

Now that you've installed everything, let's try running the main script, with all flags enabled, on the demo video.
Don't worry, if you've missed any installs, working intermediate results are cached under `.cache/`.
```bash
source venv/bin/activate
python long_exposure_fusion.py demo/lake.mp4 -m demo/lake.yaml -o demo/output --align --interpolate 2
```

You should see:
- FFMPEG decode the input video `demo/lake.mp4`
- LightGlue align and crop all frames to reference frame 0
- RIFE interpolate frames to double the framerate
- The Segment Picker UI open

In the Segment Picker UI:
- Select object 0 using the object buttons on the right. We'll only be using a single object in this demo.
We want our object to segment the sky.
- Left click on the sky to add a positive point to our object.
- Right click on the mountains to add a negative point to our object.
- Add more points if necessary to have SAM2 mask the sky out.
- Press 'space' to start mask propagation through the video.
This might take a while, stay hydrated!
- If the mask is lost during propagation, add more points on intermediate frames.
You can traverse the sequence using the scroll wheel.
- Wait for the masks to propagate through to the final frame of the sequence.
- Close the UI.

Finally, you should see two fused images be created using the maps defined in `demo/lake.yaml`.
Once the script is finished, go check out your results in `demo/output`.
Intermediate results are kept in `.cache/`.
If you'd like to rerun the entire pipeline, add the parameter `--clear-cache` to the main script to start from scratch.
