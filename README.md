# AvianSight: Bird Species Classification for Amateur Birdwatchers

AvianSight aims to encourage amateur birdwatchers by offering an ML tool that can accurately classify images of 525 different bird species; making ornithology and species identification more accessible, informative and engaging.

## TIMM Framework
The TIMM (PyTorch Image Models) library will be used as a framework to access pre-trained models to leverage transfer learning by fine-tuning model weights to suit classifying birds. This strategy not only accelerates the model's training process but also takes advantage of the robustness of models trained on extensive and diverse datasets, thereby improving accuracy and reliability.

## Data Description
A [Kaggle dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data) with images of 525 different species of birds is used to train the AvianSight species classification model. The dataset is comprised of 84,635 training images, 2,625 test images and 2,625 validation images, with each image having dimensions of 224 x 224 x 3 in a JPG format. The dataset has been cleaned from low information images, duplicates and near-duplicates, and birds often occupy more than 50% of the image area, to ensure a high-quality data foundation. The dataset is unbalanced, yet each species has at least 130 training image files, and about 80% of the images are of the colorful male species while the remaining 20% are of the more bland female. Almost all test and validation images are taken from the male of the species; with 5 images per species.

## Model Exploration
To achieve an accurate AvianSight, several pre-trained state-of-the-art models will be experimented with, including but not limited to:

- **EfficientNet**: Known for its balance between accuracy and computational efficiency, EfficientNet is an ideal model candidate considering the potential deployment on mobile devices for amateur birdwatchers.

- **CAFormer**: Integrating the strengths of attention mechanisms with convolutional networks, CAFormer is expected to excel in accurately classifying bird species as it set a new performance record on ImageNet-1K in December 2023.

___

*The project follows a [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).*
