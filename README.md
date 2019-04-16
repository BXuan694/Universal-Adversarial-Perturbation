# Universal Adversarial Perturbation
This is PyTorch version of Universal Adversarial Perturbation (https://arxiv.org/abs/1610.08401).

The code is based on the below:
https://github.com/LTS4/universal/tree/master/python
https://github.com/LTS4/DeepFool/tree/master/Python

The dataset is Caltech256 (http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

The model is ResNet50 trained on Caltech256 with accuracy about 86%. You can download from https://www.dropbox.com/sh/4xoujz0v5j8bt42/AAB_gF1fIwQ2KPA-JeAv8wqma?dl=0 and put it in ./checkpoint.

The .npy perturbation file is of 3\*224\*224, a test picture must be transformed with cut() in trainfile.py and then added with which.

## The project is already finished, but the doc need to update. When free, I'll update the doc and test the algoritm on more classifiers. If you are patient enough, you can modify for your use.:)
