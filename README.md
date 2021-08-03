# cda-project-summer2021

## Resources
*  [Skin cancer image classification dataset](https://www.kaggle.com/surajghuwalewala/ham1000-segmentation-and-classification)
*  [Loading Model Pytorch Model weight files](https://pytorch.org/tutorials/beginner/saving_loading_models.html)


## Purpose
Repository for CDA project for skin cancer image classification. The project goal is to

## Install Instructions
1. Clone Repo 
```py
git clone 
```
3. [Download Skin Lesion Dataset] (https://www.kaggle.com/surajghuwalewala/ham1000-segmentation-and-classification)
4. Unzip the 'archive.zip' and move to data folder. Use cda-project-summer2021/data/... as path in code
5. Follow the Environement Setup instructions below.


## Environement Setup
##### Run on your local machine in a virtual environment with Python3.
* git clone
* virtualenv -p python3 3envname
* source 3envname/bin/activate
* pip install -r requirements.txt
* python main.py

##### Run on your local machine with Docker.
- git clone 
- cd gtm-qa-tool
- docker build -t qa-app .
- docker run --name qa-app
