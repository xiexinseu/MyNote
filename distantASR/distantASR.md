# distantASR
[TOC]



## 工程网址如下：
https://github.com/mmdagent/distantspeechrecognition-mirror.git

## 环境准备

1. gsl - GNU Scientific Library
https://www.linuxidc.com/Linux/2016-04/129729.htm
安装了sudo apt-get install gsl-bin

2. swig - simplified wrapper and interface generator
按照该网站步骤
https://blog.csdn.net/veryitman/article/details/17398151


* 配置path出现问题。
直接打开~/.bashrc，添加
```bash
# add swig to the path
export SWIG_PATH=/home/xiexin/swigtool/bin
export PATH=$SWIG_PATH:$PATH
```

* 还是有问题参考：https://blog.csdn.net/liang12360640/article/details/41806515
sudo ln -s /usr/local/lib/libpcre.so.1 /lib


* `$ swig -version`显示
swig: error while loading shared libraries: libpcre.so.1: cannot open shared object file: No such file or directory
找不到libpcre.so.1，在路径：/lib/x86_64-linux-gnu中，只找到libpcre.so.3，所以建立链接`sudo ln -s libpcre.so.3 libpcre.so.1`

3. autoconf和automake
`sudo apt install autoconf`

4. pkgconfig - automatic project configuration
`sudo apt install pkg-config`

5. libtool - building shared object libraries
`sudo apt install libtool-bin `已有


6. Configuration问题
```bash
  cd ~/src/asr
  ./autogen.sh
  ./configure
```
发现找不到Python library path，
修改

```bash
	for i in "$python_path/lib64/python$PYTHON_VERSION/config/" "$python_path/lib64/python$PYTHON_VERSION/" "$python_path/lib64/python/config/" "$python_path/lib64/python/" "$python_path/lib/python$PYTHON_VERSION/config/" "$python_path/lib/python$PYTHON_VERSION/" "$python_path/lib/python/config/" "$python_path/lib/python/" "$python_path/" ; do
```
为
```bash
	for i in "$python_path/lib64/python$PYTHON_VERSION/-3.6m-x86_64-linux-gnu/" "$python_path/lib64/python$PYTHON_VERSION/" "$python_path/lib64/python/-3.6m-x86_64-linux-gnu/" "$python_path/lib64/python/" "$python_path/lib/python$PYTHON_VERSION/-3.6m-x86_64-linux-gnu/" "$python_path/lib/python$PYTHON_VERSION/" "$python_path/lib/python/-3.6m-x86_64-linux-gnu/" "$python_path/lib/python/" "$python_path/" ; do
```
修改
```bash
		python_path=`find $i -type f -name libpython$PYTHON_VERSION.* -print| head -n 1`
```
为
```bash
		python_path=`find $i -type f -name libpython$PYTHON_VERSIONm.* -print| head -n 1`
```
因为观察anaconda中Python的文件夹除了带版本号，还多了个m。