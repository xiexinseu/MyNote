# kaldi安装编译

<!-- toc -->
## 下载
`git clone https://github.com/kaldi-asr/kaldi.git kaldi-trunk --origin golden`

## 各目录功能
### ./tools目录
1. [OpenFST](http://www.openfst.org/twiki/bin/view/FST/WebHome)
Weighted Finite State Transducer library，是一个用来构造有限状态自动机的库。我们知道隐马尔科夫模型就可以看成是一个有限状态自动机的。这是最终要的一个包，Kaldi的文档里面说：If you ever want to understand Kaldi deeply you will need to understand OpenFst.
2. ATLAS
这是一个C++下的线性代数库。做机器学习自然是需要很多矩阵运算的。
3. [IRSTLM](https://sourceforge.net/projects/irstlm/)
这是一个统计语言模型的工具包。
4. [sph2pipe](https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools)
这是宾夕法尼亚大学linguistic data consortium（LDC）开发的一款处理SPHERE_formatted数字音频文件的软件，它可以将LDC的sph格式的文件转换成其它格式。
### ./src目录
存放的是Kaldi的源代码。
### ./egs
存放的是Kaldi提供的一些例子。我们现在要做的就是编译安装Kaldi依赖的各种库，然后编译安装Kaldi。

## Kaldi的编译
安装之前需要确保你安装了这些软件：
apt-get 
subversion 
automake 
autoconf 
libtool 
g++ 
zlib 
libatal 
wget
安装方法为在shell里输入：
```
sudo apt-get install subversion
sudo apt-get install automake
sudo apt-get install autoconf
sudo apt-get install libtool
sudo apt-get install g++
sudo apt-get install wget
sudo apt-get install libatlas-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install zlib1g
sudo apt-get install zlib1g-dev 
```
1. 在./tool目录下输入make，开始编译，输入make -j 4命令可以加快速度（多核并行处理）。
2. 之后切换到./src目录下，输入./configure进行配置，然后输入make all进行编译。当然这个过程也可以并行处理加速，输入make -j 4。经过漫长的编译过程以后，就安装完毕了。