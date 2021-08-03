# cda-project-summer2021

## Resources
*  [Skin cancer image classification dataset](https://www.kaggle.com/surajghuwalewala/ham1000-segmentation-and-classification)
*  [Loading Model Pytorch Model weight files](https://pytorch.org/tutorials/beginner/saving_loading_models.html)


## Purpose
Repository for CDA project for skin cancer image classification. The project goal is to correctly classify skin lesions with high accuracy using machine learning models.

## Install Instructions
1. Clone Repo with below command.
```py
git clone https://github.com/walnut-john/cda-project-summer2021.git
```
3. [Download Skin Lesion Dataset](https://www.kaggle.com/surajghuwalewala/ham1000-segmentation-and-classification)
4. Unzip the 'archive.zip' and move to data folder. Use cda-project-summer2021/data/... as path in code
5. Follow the Environement Setup instructions below.


## Environement Setup
##### Run on your local machine in a virtual environment with Python3.
* virtualenv -p python3 3envname
* source 3envname/bin/activate
* pip install -r requirements.txt
* jupyter lab

##### Run on your local machine with Docker.
- cd cda-project-summer2021
- docker build -t image-classifier-models .
- docker run --name image-classifier-models
