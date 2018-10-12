# AI学习

[TOC]

本文涵盖了神经网络结构、机器学习、TensorFlow、Pandas、Numpy、Python、Scikit-Learn、Scipy等的基本概念与使用方法。

来源：https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463

## 微积分
[Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced.pdf)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced1.png)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced2.png)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced3.png)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced4.png)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced5.png)

![Calculus_Cheat_Sheet_All_Reduced](Calculus_Cheat_Sheet_All_Reduced6.png)

## 概率论
[probability_cheatsheet](probability_cheatsheet.pdf)

![probability_cheatsheet1](probability_cheatsheet1.png)

![probability_cheatsheet2](probability_cheatsheet2.png)

![probability_cheatsheet3](probability_cheatsheet3.png)

![probability_cheatsheet4](probability_cheatsheet4.png)

![probability_cheatsheet5](probability_cheatsheet5.png)

![probability_cheatsheet6](probability_cheatsheet6.png)

![probability_cheatsheet7](probability_cheatsheet7.png)

![probability_cheatsheet8](probability_cheatsheet8.png)

![probability_cheatsheet9](probability_cheatsheet9.png)

![probability_cheatsheet10](probability_cheatsheet10.png)

## 统计学
[stats_handout](stats_handout.pdf)

![stats_handout](stats_handout1.png)

![stats_handout](stats_handout2.png)

![stats_handout](stats_handout3.png)

![stats_handout](stats_handout4.png)

![stats_handout](stats_handout5.png)

![stats_handout](stats_handout6.png)

![stats_handout](stats_handout7.png)

![stats_handout](stats_handout8.png)

## 线性代数
[linear_algebra_in_4_pages](linear_algebra_in_4_pages.pdf)

![linear_algebra_in_4_pages1](linear_algebra_in_4_pages1.png)

![linear_algebra_in_4_pages2](linear_algebra_in_4_pages2.png)

![linear_algebra_in_4_pages3](linear_algebra_in_4_pages3.png)

![linear_algebra_in_4_pages4](linear_algebra_in_4_pages4.png)



## 神经网络

![神经网络](神经网络.png)
## 神经网络图
![神经网络图](神经网络图.png)
![神经网络公式1](神经网络公式1.png)
![神经网络公式2](神经网络公式2.png)
## 机器学习概览
![机器学习概览](机器学习概览.png)
## 机器学习：Scikit-learn算法
这个部分展示了Scikit-learn中每个算法的适用范围及优缺点，可以帮你快速找到解决问题的方法。
![Scikit-learn框图](Scikit-learn框图.png)
## Scikit-learn

Scikit-learn（以前称为scikits.learn）是机器学习库。 它具有各种分类，回归和聚类算法，包括支持向量机，随机森林，梯度增强，k-means和DBSCAN等。
![Scikit-learn](Scikit-learn.png)

[Python数据科学速查表-Scikit-Learn](中文-pdf/Python数据科学速查表-Scikit-Learn.pdf)

```python
# Step 1: Importing the libraries
import numpy as np
import pandas as pd
# Step 2: Importing dataset
dataset = pd.read_csv('./datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
# Step 3: Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
# Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
# Step 5: Splitting the datasets into training sets and Test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
# Step 6: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```

**OneHotEncoder解释**

```python
import numpy as np 
from sklearn.preprocessing import OneHotEncoder 
enc = OneHotEncoder() 
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]]) 
print "enc.n_values_ is:",enc.n_values_
print "enc.feature_indices_ is:",enc.feature_indices_
print enc.transform([[0, 1, 1]]).toarray()

enc.n_values_ is: [2 3 4]
enc.feature_indices_ is: [0 2 5 9]
[[ 1. 0. 0. 1. 0. 0. 1. 0. 0.]]
```

**首先由四个样本数据[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]，共有三个属性特征，也就是三列。比如第一列，有0,1两个属性值，第二列有0,1,2三个值.....，那么enc.n_values_就是每个属性列不同属性值的个数，所以分别是2,3,4。再看enc.feature_indices_是对enc.n_values_的一个累加。再看[0, 1, 1]这个样本是如何转换为基于上面四个数据下的one-hot编码的。第一列：0->10；第二列：1->010；第三列：1->0100。简单解释一下，在第三列有，0,1,2,3四个值，分别对应1000,0100,0010,0001.**

![Python数据科学速查表-Scikit-Learn](Python数据科学速查表-Scikit-Learn.png)

## 监督和非监督算法公式集
[监督和非监督算法公式集](监督和非监督算法公式集.pdf)

## 机器学习：算法
Microsoft Azure的这款机器学习备忘单将帮助您为预测分析解决方案选择合适的机器学习算法。

![机器学习备忘单](机器学习备忘单.png)

## 机器学习：算法(Python&R codes)

![机器学习算法Python&RCodes](机器学习算法Python&RCodes.jpg)

## 数据科学中的Python

[Python数据科学速查表-Python基础](中文-pdf/Python数据科学速查表-Python基础.pdf)

![Python数据科学速查表-Python基础](Python数据科学速查表-Python基础.png)

![python数据科学](python数据科学.png)

[python基础](python基础.pdf)

![python基础1](python基础1.png)

![python基础2](python基础2.png)

## Bokeh
[Python数据科学速查表-Bokeh](中文-pdf/Python数据科学速查表-Bokeh.pdf)

![Python数据科学速查表-Bokeh](Python数据科学速查表-Bokeh.png)

![python大数据](python大数据.png)

## Python数据科学速查表-JupyterNotebook
[Python数据科学速查表-JupyterNotebook](中文-pdf/Python数据科学速查表-JupyterNotebook.pdf)

![Python数据科学速查表-JupyterNotebook](Python数据科学速查表-JupyterNotebook.png)

## TensorFlow
![TensorFlow](tensorflow.png)

## Keras
2017年，Google的TensorFlow团队决定在TensorFlow的核心库中支持Keras。 Chollet解释说，Keras被认为是一个界面而不是端到端的机器学习框架。 它提供了更高级别，更直观的抽象集，无论后端科学计算库如何，都可以轻松配置神经网络。

![keras](keras.jpeg)

[Python数据科学速查表-Keras](中文-pdf/Python数据科学速查表-Keras.pdf)

![Python数据科学速查表-Keras](Python数据科学速查表-Keras.png)

## NumPy
NumPy通过提供多维数组以及在数组上高效运行的函数和运算符来提高运算效率，需要重写一些代码，主要是使用NumPy的内部循环。
![NumPy](NumPy.png)

[Python数据科学速查表-Numpy基础](中文-pdf/Python数据科学速查表-Numpy基础.pdf)

![Python数据科学速查表-Numpy基础](Python数据科学速查表-Numpy基础.png)

[numpy-cheat-sheet](numpy-cheat-sheet.pdf)

![numpy-cheat-sheet](numpy-cheat-sheet.png)

[numpy-cheat-sheet2](numpy-cheat-sheet2.pdf)

![numpy-cheat-sheet2](numpy-cheat-sheet2.png)
## Pandas
“Pandas”这个名称来自术语““panel data ”，这是一个多维结构化数据集的计量经济学术语。
![Pandas](Pandas.png)

[pandas](pandas.pdf)

![Pandas1](Pandas1.png)

![Pandas2](Pandas2.png)

![Pandas3](Pandas3.png)

![Pandas4](Pandas4.png)

[Python数据科学速查表-Pandas基础](中文-pdf/Python数据科学速查表-Pandas基础.pdf)

![Python数据科学速查表-Pandas基础](Python数据科学速查表-Pandas基础.png)

[Python数据科学速查表-Pandas进阶](中文-pdf/Python数据科学速查表-Pandas进阶.pdf)

![Python数据科学速查表-Pandas进阶](Python数据科学速查表-Pandas进阶.png)

## 数据清洗
Data Wrangling 是一款好用的数据清洗软件。
![数据清洗](数据清洗.jpeg)
![数据清洗2](数据清洗2.jpeg)
## dplyr 和tidyr的数据清洗
![dplyr&tidyr数据清洗](dplyr&tidyr数据清洗.jpeg)
![dplyr&tidyr数据清洗2](dplyr&tidyr数据清洗2.jpeg)
## SciPy
SciPy建立在NumPy数组对象之上，是NumPy工具集的一部分。
![SciPy](SciPy.png)
## Matplotlib
![Matplotlib](Matplotlib.png)

[Python数据科学速查表-Matplotlib绘图](中文-pdf/Python数据科学速查表-Matplotlib绘图.pdf)

![Python数据科学速查表-Matplotlib绘图](Python数据科学速查表-Matplotlib绘图.png)

## Python数据科学速查表-Seaborn
[Python数据科学速查表-Seaborn](中文-pdf/Python数据科学速查表-Seaborn.pdf)

![Python数据科学速查表-Seaborn](Python数据科学速查表-Seaborn.png)

## Python数据科学速查表-导入数据
[Python数据科学速查表-导入数据](中文-pdf/Python数据科学速查表-导入数据.pdf)

![Python数据科学速查表-导入数据](Python数据科学速查表-导入数据.png)

## 数据可视化
![数据可视化](数据可视化.jpeg)
![数据可视化2](数据可视化2.jpeg)
## PySpark
![PySpark](PySpark.jpeg)
## Big-O
各种算法的复杂度
![Big-O](Big-O.png)
![Big-O复杂度](Big-O复杂度.png)
![数据结构](数据结构.png)
![排序算法复杂度](排序算法复杂度.png)
## 参考资料
Big-O Algorithm Cheat Sheet: http://bigocheatsheet.com/

Bokeh Cheat Sheet: https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Bokeh_Cheat_Sheet.pdf

Data Science Cheat Sheet: https://www.datacamp.com/community/tutorials/python-data-science-cheat-sheet-basics

Data Wrangling Cheat Sheet: https://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf

Data Wrangling: https://en.wikipedia.org/wiki/Data_wrangling

Ggplot Cheat Sheet: https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf

Keras Cheat Sheet: https://www.datacamp.com/community/blog/keras-cheat-sheet#gs.DRKeNMs

Keras: https://en.wikipedia.org/wiki/Keras

Machine Learning Cheat Sheet: https://ai.icymi.email/new-machinelearning-cheat-sheet-by-emily-barry-abdsc/

Machine Learning Cheat Sheet: https://docs.microsoft.com/en-in/azure/machine-learning/machine-learning-algorithm-cheat-sheet

ML Cheat Sheet:: http://peekaboo-vision.blogspot.com/2013/01/machine-learning-cheat-sheet-for-scikit.html

Matplotlib Cheat Sheet: https://www.datacamp.com/community/blog/python-matplotlib-cheat-sheet#gs.uEKySpY

Matpotlib: https://en.wikipedia.org/wiki/Matplotlib

Neural Networks Cheat Sheet: http://www.asimovinstitute.org/neural-network-zoo/

Neural Networks Graph Cheat Sheet: http://www.asimovinstitute.org/blog/

Neural Networks: https://www.quora.com/Where-can-find-a-cheat-sheet-for-neural-network

Numpy Cheat Sheet: https://www.datacamp.com/community/blog/python-numpy-cheat-sheet#gs.AK5ZBgE

NumPy: https://en.wikipedia.org/wiki/NumPy

Pandas Cheat Sheet: https://www.datacamp.com/community/blog/python-pandas-cheat-sheet#gs.oundfxM

Pandas: https://en.wikipedia.org/wiki/Pandas_(software)

Pandas Cheat Sheet: https://www.datacamp.com/community/blog/pandas-cheat-sheet-python#gs.HPFoRIc

Pyspark Cheat Sheet: https://www.datacamp.com/community/blog/pyspark-cheat-sheet-python#gs.L=J1zxQ

Scikit Cheat Sheet: https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet

Scikit-learn: https://en.wikipedia.org/wiki/Scikit-learn

Scikit-learn Cheat Sheet: http://peekaboo-vision.blogspot.com/2013/01/machine-learning-cheat-sheet-for-scikit.html

Scipy Cheat Sheet: https://www.datacamp.com/community/blog/python-scipy-cheat-sheet#gs.JDSg3OI

SciPy: https://en.wikipedia.org/wiki/SciPy

TesorFlow Cheat Sheet: https://www.altoros.com/tensorflow-cheat-sheet.html

Tensor Flow: https://en.wikipedia.org/wiki/TensorFlow

##### 