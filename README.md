<h1 align="center">
    Img2brain
</h1>

<p align="center">
    <a href="https://github.com/sayalaruano/NTDs2RDF/blob/main/LICENSE.md">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/bioregistry" />
    </a>
        <a href="https://doi.org/10.5281/zenodo.7979730">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7979730.svg" alt="DOI">
    </a>
    <a href="https://colab.research.google.com/github/sayalaruano/img2brain/blob/main/EDA_feateng_modelbuild_img2brain.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab">
    </a>
</p>

<p align="center">
   Predicting the neural responses to visual stimuli of naturalistic scenes using ML-based visual encoding models.
</p>

## Table of contents:

- [About the project](#about-the-project)
- [Dataset](#dataset)
- [Feature engineering](#feature-engineering)
- [Machine learning models](#machine-learning-models)
- [How to set up the environment to run the code?](#how-to-set-up-the-environmen-to-run-the-code)
- [Structure of the repository](#structure-of-the-repository)
- [Credits](#credits)
- [Further details](#details)
- [Contact](#contact)

## About the project

Vision is one of the main sensory pathways that enable living organisms to perceive external stimuli and interpret
the world. The **human visual system (HVS)** involves a complex interplay between the eyes, the brain, and multiple
neural pathways. To study the HVS, investigators employ various experimental methods such as psychophysics, eye tracking, 
and neuroimaging techniques like **functional magnetic resonance imaging (fMRI)**. fMRI uses magnetic resonance
imaging to measure the **blood-oxygen-level-dependent (BOLD) variation** induced by neuronal activity.

The data acquired through fMRI experiments are stored in the form of **voxels**, which are three-dimensional units
representing tiny volume elements. Each voxel encompasses millions of brain cells, collectively forming areas with 
different functional properties, known as **regions of interest (ROIs)**. **Computational models** are essential tools for 
interpreting the large datasets produced by fMRI experiments. 

The goal of this project is to employ **machine learning** techniques to **forecast the neural visual responses triggered by naturalistic scenes**. These computational models aim to **replicate the process through which neuronal activity encodes to visual stimuli** 
aroused by the external environment. The following figure gives an schematic representation of the **brain encoding and decoding processes**.

<p align="center">
<figure>
  <img src="./img/encod_decod_brain.png" alt="my alt text"/>
  <figcaption> Figure 1. Brain encoding and decoding in fMR. Obtained from <a href="https://doi.org/10.1016/j.eng.2019.03.010">[1]</a>. </figcaption>
</figure>
</p>

**Visual encoding models** based on fMRI data employ algorithms that transform image pixels into model features and map these features to brain activity. This framework enables the **prediction of neural responses from images**. The following figure illustrates the mapping between the pixel, feature, and brain spaces.

<p align="center">
<figure align="center">
  <img src="./img/encod_brain_sp.png" alt="my alt text"/>
  <figcaption>Figure 2. The general architecture of visual encoding models that consists of three spaces (the input space, the feature space, and the brain activity space) and two in-between mappings. Obtained from <a href="https://doi.org/10.1016/j.neuroimage.2010.07.073">[2]</a>. </figcaption>
</figure>
</p>

## Dataset

The data for this project is part of the [Natural Scenes Dataset][nsd] (NSD), a massive **dataset of 7T fMRI responses to images of natural scenes** coming from the [COCO dataset][coco]. The **training dataset** consists of **brain responses measured at 10.000 brain locations (voxels) to 8857 images** (in jpg format) for one subject. The 10.000 voxels are distributed around the visual pathway and may encode **perceptual and semantic features** in different proportions. The **test dataset** comprises **984 images** (in jpg format), and the goal is to predict the brain responses to these images.

You can access the dataset through Zenodo with the following DOI: [doi.org/10.5281/zenodo.7979729][dataset_doi].

The training dataset was split into **training and validation partitions** with an **80/20 ratio**. The training partition 
was used to train the models, and the validation partition was used to evaluate the models. The test dataset was used to make 
predictions with the best model on unseen data.

The figure below summarizes the data splitting process and the rest of the steps to build the visual encoding models.

<p align="center">
<figure align="center">
  <img src="./img/encod_model_scheme.png" alt="my alt text"/>
  <figcaption>Figure 3. Summary of the main steps for the creation of the visual encoding models. Obtained from <a href="https://doi.org/10.1016/j.neuroimage.2010.07.073">[2]</a>. </figcaption>
</figure>
</p>

## Feature engineering

Due to the **high dimensionality of the feature representation of images** using the raw pixel values (i.e., the original images have a size of 425x425 and 3 channels (RGB), which results in a feature representation of 425x425x3 = 541875 features), I used the **representations obtained from different layers of pretrained CNNs** to obtain a lower dimensional representation of the images. In this case, I tried various layers of four different pretrained CNNs: [AlexNet][alexnet], [VGG16][vgg16], [ResNet50][resnet50], and [InceptionV3][inceptionv3], available in the [torchvision package][torchvision].

A graphical representation of the feature engineering process is presented below.

<p align="center">
<figure align="center">
  <img src="./img/linearizing_encoding_algorithm.jpg" alt="my alt text" width="500"/>
  <figcaption>Figure 4. Diagram of the feature engineering stage. Obtained from <a href="https://doi.org/10.48550/arXiv.2301.03198">[3]</a>. </figcaption>
</figure>
</p>

The **feature representations** of the images was obtained by passing the images through the pretrained CNNs and extracting the output of the desired layer. The size of the feature vectors at this point was still very large, so I used **PCA** to overcome this problem and got a **set of 30 features**. I fit the PCA on the training images features, and used it to downsample the training, validation and test images features. 

I evaluated the best feature representation by **training a simple linear regression model to predict the brain activity** of the voxels from the feature representation of the images. The **best feature representation** was the one that resulted in the **highest encoding accuracy** (i.e., median correlation between the predicted and actual brain activity of the voxels) on the validation set.

You can find the code and results for this part of the project [here][notebook_feateng].

## Machine learning models

I trained **6 different machine learnning algorithms** (linear regression - base model, ridge regression, lasso regression, elasticnet regression, k-nearest neighbours regressor, and decision tree regressor) to **predict the brain activity of the voxels from the feature representation of the images**. In this project, the learning task was a **multioutput regression problem**, where the **input is the feature representation of the images and the output is the brain activity of all the voxels**. Each regressor maps from the feature space to each voxel, so there is a separate encoding model per voxel, leading to **voxelwise encoding models**. Therefore, **every model trained with this dataset have 10.000 independently regression models with n coeficients each** (the number of features). As in the previous section, the best model was the one that resulted in the highest encoding accuracy on the validation set.

## Results of the ML models

The best model was **Lasso regression** with an encoding accuracy of **0.2417** on the validation set. The best hyperparameters were `alpha=0.01` and the default `max_iter=1000`. This model was trained with the feature representation of the images obtained from layer `features.12` of the **AlexNet CNN**, reduced to **100 features** using PCA.

| Machine Learning Model | Encoding Accuracy |
| :--- | :---: |
| **Lasso (alpha=0.01)** | **0.2417** |
| ElasticNet (alpha=0.001) | 0.2415 |
| Ridge (alpha=1.0) | 0.2412 |
| Linear Regression | 0.2402 |
| K-Nearest Neighbors | 0.1021 |
| Decision Tree | 0.0382 |

While the overall encoding accuracy of the best model is low, the distribution of predictions across voxels reveals important patterns: regularized linear models show **right-skewed accuracy distributions** with a **heavy tail between 0.4-0.7**, indicating that the **models make accurate predictions on a subset of voxels while struggling with others**. The lack of ROI information prevents us from identifying which visual areas correspond to high or low predictions.

<br />
<p align="center">
    <img src="./img/Corrcoef_allmodels.png" width="800" alt="Encoding_Accuracy_Distribution" width="500"/>
    <br />
    <em><b>Figure 5.</b> Histograms of encoding accuracy for machine learning models across all voxels. Models were trained to predict neural responses to visual stimuli from naturalistic images.</em>
</p>
<br />

To validate these findings, we **compared predictions from the best and worst performing voxels**. The top-performing voxel showed strong agreement between predicted and actual BOLD signals with high positive correlation, while the lowest-performing voxel showed no meaningful overlap or correlation pattern. This demonstrates that the **model successfully captures neural responses for a subset of voxels** but **fails to generalize across the entire brain**.

<br />
<p align="center">
    <img src="./img/fmri_pred_actual.png" width="800" alt="BOLD_Predictions"/>
    <br />
    <em><b>Figure 6.</b> Predicted vs. actual BOLD variation for the best (A-B) and worst (C-D) performing voxels from the Lasso regression model.</em>
</p>
<br />

Check it out the code for this part of the project [here][notebook_ml].

## How to set up the environmen to run the code?

I used [conda][conda] to create a virtual environment with the required libraries to run the code. To create a Python virtual environment with libraries and dependencies required for this project, you should clone this GitHub repository, open a terminal, move to the folder containing this repository, and create a conda virtual environment with the following commands:

```bash
# Create the conda virtual environment
$ conda env create -f img2brain_env.yml

# Activate the conda virtual environment
$ conda activate img2brain_env
```
Then, you can open the Jupyter notebook with the IDE of your choice and run the code.

## Structure of the repository

The main files and directories of this repository are:

|File|Description|
|:-:|---|
|[EDA_feateng_modelbuild_img2brain.ipynb](EDA_feateng_modelbuild_img2brain.ipynb)|Jupyter notebook with EDA, feature engineering, creation of the machine learning algorithms, performance metrics of all models, and evaluation of the best model|
|[LassoRegressor_alpha0.01_img2brain.bin](LassoRegressor_alpha0.01_img2brain.bin)|Bin file of the best model|
|[img2brain_env.yml](img2brain_env.yml)|File with libraries and dependencies to create the conda virtual environment|
|[img2brain_report.pdf](img2brain_report.pdf)|Report with detailed explanation of the project|
|[Results/](Results/)|Folder to save performance metrics and other outputs of the machine learning models|
|[Scripts_plots/](Scripts_plots/)|Folder for the scripts to create the plots of the report|
|[img/](img/)|images and gifs|

## Credits

- Developed by [Sebastián Ayala Ruano](https://sayalaruano.github.io/). I created this project for my capstone project of the Machine learning course from the [MSc in Systems Biology][sysbio] at [Maastricht University][maasuni].

- Part of the code was inspired by the [Algonauts Project 2023 Challenge][alg_web] development kit tutorial.

## Further details

More details about the biological background of the project, the interpretation of the results, and ideas for further work are available in this [pdf report](img2brain_report.pdf).

## Citation 

**Ayala-Ruano, S.** (2023). **Img2brain: Predicting the neural responses to visual stimuli of naturalistic scenes using machine learning** (Version 1.0.0) [Dataset/Software]. *Zenodo*. doi: [doi.org/10.5281/zenodo.7979729][data-arch].

## Contact

If you have comments or suggestions about this project, you can [open an issue][issues] in this repository.

[nsd]: https://doi.org/10.1038/s41593-021-00962-x
[coco]: https://cocodataset.org/#home
[alg_web]: http://algonauts.csail.mit.edu
[sysbio]: https://www.maastrichtuniversity.nl/education/master/systems-biology
[maasuni]: https://www.maastrichtuniversity.nl/
[dataset_doi]: https://doi.org/10.5281/zenodo.7979729
[alexnet]: https://pytorch.org/vision/master/models/alexnet.html
[vgg16]: https://pytorch.org/vision/master/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16
[resnet50]: https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
[inceptionv3]: https://pytorch.org/vision/master/models/inception.html
[torchvision]: https://pytorch.org/vision/stable/index.html
[notebook_feateng]: ./EDA_feateng_modelbuild_img2brain.ipynb#3-feature-engineering
[notebook_ml]: ./EDA_feateng_modelbuild_img2brain.ipynb#4-bulding-and-evaluating-machine-learning-models
[conda]: https://docs.conda.io/en/latest/
[issues]: https://github.com/sayalaruano/img2brain/issues/new