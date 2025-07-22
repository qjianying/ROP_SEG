# ROP_SEG
## Prerequisites
python3.7  
torch1.10.1+cu113  
torchvision0.11.2+cu113  
easydict1.9  
tqdm4.64.0  
## Training
1、Create the config file of dataset:train.txt, val.txt and test.txt,the file structure is as follows:  
``path-of-the-image   path-of-the-groundtruth``  
Refer to ``furnace/tools/generate_source.py`` for reference  
2、Modify the config.py according to your requirements  
3、Train a network  
### Distributed Training  
We use the official torch.distributed.launch in order to launch multi-gpu training. This utility function from PyTorch spawns as many Python processes as the number of GPUs we want to use, and each Python process will only use a single GPU.  
For each experiment, you can just run this script:  
```export NGPUS=2```  
```python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py```  
``python eval.py``
## Inference
