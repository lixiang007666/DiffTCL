<p align="center"><img src="imgs/difftcl_logo.png"/ width="100%"><br></p>

This is the official repository for the paper: [Diffusion-Enhanced Transformation Consistency Learning for Retinal Image Segmentation](#). 
 

## News
Here is the latest news:
- :rocket: The first version of the code has been released, and a complete, optimized version of DiffTCL will be released shortly.
- :rocket: The paper is in submission.
- :rocket: WIP.

## Overview 

> [!NOTE]
The abstract graphic will be published shortly.

## Quick Start

### ⚙️ Installation
```shell
conda create -n difftcl python=3.8.16
pip install -r requirements
```

### 📉 Training
```python
# Diffusion-Enhanced pre-training
python3 train.py --config configs/diff-pre-training.yaml
python3 scripts/extract_model_weights.py -c path/to/checkpoint/file
# TCL
sh scripts/train.sh 4 <port>
```


## BibTeX
TODO
