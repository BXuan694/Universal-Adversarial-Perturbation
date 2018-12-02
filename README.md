# UniversalAdversarialPerturbation
This is PyTorch version of Universal Adversarial Perturbation(https://arxiv.org/abs/1610.08401)

The code is based on the below:
https://github.com/LTS4/universal/tree/master/python
https://github.com/LTS4/DeepFool/tree/master/Python

The dataset is based on Caltech256(http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

The model is ResNet50 trained on Caltech256 with accuracy about 96%. Some other models will be tested in the future.

The .npy perturbation file is of 3*224*224, a test picture must be transformed with cut() in trainfile.py and then added with which.

## The project is not finished yet:), but it will not take a long time:)
