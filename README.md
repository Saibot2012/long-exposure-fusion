# Long-Exposure Fusion

## Introduction

**Long-Exposure Fusion** is a full Python pipeline that generates long-exposure style images from videos or burst photo sequences. Images are generated using our own variant of [Exposure Fusion](https://ieeexplore.ieee.org/document/4392748) that we call Hybrid Weight-Map Fusion.

This repository provides:
- An automated pipeline for decoding, aligning, interpolating, defining segmentation masks, and fusing images.
- A simple GUI for segmenting images.
- Command-line tools for processing your own videos or image bursts.
