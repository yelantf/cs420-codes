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

## Deep Learning Methods

### Deep Model Performance

|Model|Accuracy|Model|Accuracy|
|:---|---|:---|---|
|FC|90.11|CNN|99.47|
|FC+Largest CC| 92.46|CNN+Largest CC|99.31|
|FC+SegNet| 92.78|CNN+SegNet|99.40|
|FC+CC Centralization|99.03|CNN+CC Centralization|99.88|
|FC+LocNet|**99.28**|CNN+LocNet|**99.90**|

### SegNet Result Visualization

<div align=center>
<img src="./deep_learning_methods/img/segnet_vis.png" width="600" />
</div>

### LocNet Result Visualization

<div align=center>
<img src="./deep_learning_methods/img/locnet_vis.png" width="600" />
</div>