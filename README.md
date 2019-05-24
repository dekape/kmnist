<img src="http://codh.rois.ac.jp/img/kmnist.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

# ACSE-8-Mini-Project

## Introduction
The aim of this project was to create and implement a neural network to maximise the prediction accuracy of Kuzushiji characters using the KMNIST dataset. A 'SupervisedLearning' class was created as a supervised training wrapper over 'pytorch' and 'scikitlearn' library modules. Modified versions of the well established LetNet5 and AlexNet networks have been implemented as examples.

## Repo Structure
- KMNIST_Learning.py : python modules containing all useful classes and functions used throughout the project
- KNIST_Layout.ipynb : notebook with a laid out example of how to use the class to perform basic supervised trainings
- KMNIST_Layout_deepAlex.ipynb: notebook with tests over different versions of the AlexNet network
- KMNIST_SDeepAlex_old_aug.ipynb: notebook with tests over a deeper version of the AlexNet network
- KMNIST_TLDataAug.ipynb: notebook with examples of how to perform transfer learning with the 'SupervisedLearning' class as well as an implementation of pre processing data augmentation using the 'albumentation' libray.
- weight_decay_tuning.ipynb: notebook with tests over different weight decays for the L2 normalisation on the modified AlexNet networks

## Requirements
To be able to run this software, the following packages and versions are required:
- numpy >= 1.15.4
- matplotlib >= 3.0.2
- mpltools >= 0.2.0
- pandas >= 0.23.4
- pytorch >= 1.0.2
- scikitlearn >= 0.1.1

If running on a Colab notebook, the example given in 'KNIST_Layout.ipynb' shows how to obtain these packages.


## Usage/Examples
See 'KNIST_Layout.ipynb' for installation and usage examples.
