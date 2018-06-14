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

## Traditonal Methods

|  | Naive Bayes | Desision Tree | Random Forest | K-Nearest Neighbor | SVM |
| :----: |:------------:| :----: |:------------:| :-: | :-: |
| Target dataset | 18.81% | 50.94% | 87.61% | 88.63% | 87.07% |
| Keep largest CC | 19.73% | 55.54% | 89.07% | 88.73% |88.29%|
| Shift CC to center | 75.90% | 92.69% | 98.46% | 97.55% | 96.85%|

## Deep Learning Methods

### Deep Model Performance

||Baseline|Largest CC|CC Centralization|SegNet|LocNet|
|:---:|:---:|:---:|:---:|:---:|:---:|
|FC|90.11|92.46|92.78|99.03|**99.28**|
|CNN|99.47|99.31|99.40|99.88|**99.90**|
|PointNet|91.02|

### SegNet Result Visualization

<div align=center>
<img src="./img/segnet_vis.png" width="600" />
</div>

### LocNet Result Visualization

<div align=center>
<img src="./img/locnet_vis.png" width="600" />
</div>