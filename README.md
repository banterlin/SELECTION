This repository provides the code for [A deep learning model for personalized intra-arterial therapy planning in unresectable hepatocellular carcinom]. Based on the code, you can easily train your own SELECTION by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## A deep learning model for personalized intra-arterial therapy planning in unresectable hepatocellular carcinoma: a multicenter retrospective study
## Lin, Xiaoqi et al.
## eClinicalMedicine, Volume 75, 102808

## Overview
SELECTION is a ransformer-based multi-modal medical prognosis model. It differs from the traditional multi-modal apporach of fusing features from CNN with clinical data to classify clinical endpoint, where SELECTION considers holistic multi-modal information from CECT images as well as clinical information as sequences of tokens.

## Setup the Environment
This software was implemented a system running `Windows 10`, with `Python 3.9`, `PyTorch 2.0.1`, and `CUDA 11.7.1`.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description
The main architecture of SELECTION lies in the `models/` folder. The `modeling_selection.py` is the main backbone, while the rest necessary modules are distributed into different files based on their own functions, i.e., `attention.py`, `block.py`, `configs.py`, `embed.py`, `encoder.py`, and `mlp.py`. Please refer to each file to acquire more implementation details. 

**Parameter description**:

`--CLS`: number of classification (binary in this study)

`--BSZ`: batch size.

`--DATA_DIR`: Folder path for the imaging data. arranged to have portal and arterial subfolers inside.

`--SET_TYPE`: file name of the clinical baseline data (`***.pkl`).

Note that `xxx.pkl` is a dictionary that stores the clinical textual data in the format of `key-value`. Here is a short piece of code showing how to organize the `***.pkl`:
```python
>>> import pickle
>>> f = open('***.pkl', 'rb')
>>> subset = pickle.load(f)
>>> f.close()
>>> list(subset.keys())[0:10] # display top 10 case ids
>>> key = list(subset.keys()) # select the patient ID
>>> subset[key] # display the clinical data
>>> subset[key]['Baseline'] # the demographics information (age and sex)
>>> subset[key]['Lab'] #  the laboratory test results
>>> subset[key]['label'] # the clinical endpoint labels
```

The code used in this studied were adopted from [A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics,doi: 10.1038/s41551-023-01045-x]

