# Semi-Supervised Learning With Soft AUC Loss For BirdCLEF2025


## Runtime details  
CPU: AMD Ryzen 7 (8 cores)  
GPU: RTX 4090  
OS: Linux  
Python version: 3.10   
Pytorch version: 2.5.1  

## Setup  
1. Install Python 3.10 and Pytroch 2.5.1+CUDA if you don't have them on your machine.  
How to install Python: https://docs.python.org/3/using/unix.html#on-linux  
How to install Pytroch: https://pytorch.org/get-started/previous-versions/  

2. Install the required Python packages:  
`pip install -r requirements.txt`  

## Run the code 
Frist of all, set data paths in "SETTINGS.json".  

Training steps:    
1. Run 'python prepare_data.py'.   
When it's finised, you can find all prepared data under the path "data_cache_dir" ("data_cache/" by default) set in "SETTINGS.json".   

2. Run 'python train.py'.   
When it's finised, you can find all trained models under the setting "model_dir" ("models/" by default) specified in "SETTINGS.json".   
   
Inference steps:   
Run 'python predict.py'.   
When it's finised, you can find the submission csv file under the setting "submission_dir" ("output/" by default) specified in "SETTINGS.json".  

## Some tips
Since "./models" contains all trained models, you can directly run 'python predict.py' without running the traning steps.   
All code-generated files will automatically overwrite existing files, so make sure to copy "./models" before you run the training steps if you want to keep the original models.   
The data preparation code and training code are composed of separate steps. You can split the parts of running data preparation, pre-training and training according to the format of "prepare_data.py" and "train.py".  



