# CardiCat
### CardiCat: a Variational Autoencoder for High-Cardinality Tabular Data
* * *   
## Overview
```
CardiCat, a general variational autoencoder (VAE) model, can
accurately fit imbalanced high-cardinality and heterogeneous tabular
data. Our method substitutes one-hot encoding with embedding layers
for high categorical variables, which are learned jointly at the encoder
and decoder levels via the generative model's loss. 
```  
* * *  
 
## Requirements 

CardiCat has been developed and runs on Python 3.9.

The usage of a virtualenv is highly recommended in order for all the dependencies and library to work well.

* * *   

## Installation

The simplest and recommended way is to follow these steps:

### For CONDA support:
conda create -n CardiCat_neurips python=3.9
conda activate CardiCat_neurips
pip install -r [USER PATH]/requirements.txt

### For python's virtualenv:
virtualenv -p python39 CardiCat_iclr
source CardiCat_neurips/bin/activate # for mac/linux
.\CardiCat_iclr\Scripts\activate # for windows
pip install -r [USER PATH]/requirements.txt

### If you'd like to connect the CardiCat virtual env kernel to your main jupyter lab/notebook:
ipython kernel install --user --name=CardiCat_iclr # adding the kernel to jupyter notebook/lab

* * *   
