# CS420 Machine Learning, Final Project
Classification on modified MNIST dataset

## Requirements
- [NumPy](https://github.com/numpy/numpy)
- [Pillow](https://github.com/python-pillow/Pillow)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [scikit-learn](http://scikit-learn.org/stable/index.html)
- [PyTorch](https://github.com/pytorch/pytorch) [0.4]

## Prepare Data
Download datasets from [jbox](https://jbox.sjtu.edu.cn/l/VooiCd) and move them to [mnist](./mnist) folder, the folder structure should look like this:

    ---- mnist/
        ---- mnist_train/
        ---- mnist_test/

## Tratditonal Methods

### Naive Bayes

```shell
cd traditional_methods/NaiveBayes/;
python Bayes.py
```

### Decision Tree

```shell
cd traditional_methods/DecisionTree/;
python Tree.py
```

### Random Forest

```shell
cd traditional_methods/RandomForest/;
python ForestBestN.py
```

These commands will output the performance of random forest with different number of decision trees, demonstrated by the following two figures.

<figure class="half">
    <img src="img/forest-accu.png">
    <img src="img/forest-runtime.png">
</figure>

### K-Nearest Neighbors

```shell
cd traditional_methods/KNN/;
python KNNBestK.py
```

These commands will output the performance of KNN with different K, demonstrated by the following figure.

<figure>
    <img src="img/k-accu.png">
</figure> 

### Support Vector Machine

```shell
cd traditional_methods/SVM/;
python SVMBestDim.py
```

These commands will output the performance of linear SVM on different dimension data reduced by PCA, demonstrated by the 
following two figures.

<figure class="half">
    <img src="img/pca-accu.png">
    <img src="img/pca-runtime.png">
</figure>

```shell
cd traditional_methods/SVM/;
python SVMBestKernel.py
```

These commands will output the performance of SVM with different kernels.

### Influence of Modification

For five traditional models above, running `*Preprocess.py` in their respective directory will give 
the results as the following table shows.

|  | Naive Bayes | Desision Tree | Random Forest | K-Nearest Neighbor | SVM |
| :----: |:------------:| :----: |:------------:| :-: | :-: |
| Target dataset | 18.81% | 50.94% | 87.61% | 88.63% | 87.07% |
| Keep largest CC | 19.73% | 55.54% | 89.07% | 88.73% |88.29%|
| Shift CC to center | 75.90% | 92.69% | 98.46% | 97.55% | 96.85%|

## Deep Learning Methods

### SegNet Result Visualization

<div align=center>

<img src="./deep_learning_methods/img/segnet_vis.png" width="600" />

![](./deep_learning_methods/img/segnet_vis.png)

</div>

### LocNet Result Visualization

<div align=center>

![](./deep_learning_methods/img/locnet_vis.png)

</div>