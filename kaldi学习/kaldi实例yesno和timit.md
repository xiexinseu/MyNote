# kaldi实例yesno和timit

[TOC]

Kaldi的例子有很多，在egs目录下面，对Kaldi不熟悉的小白们可以先从yesno和timit两个例子入手，这样可以对Kaldi有个直观的认识。

## 运行yesno实例
该实例是一个非常小的数据集，每一条记录都是一系列yes或者no的语音，标注是由文件名来标注的。先运行一下。
切换到./egs/yesno/s5目录下，运行`sudo./run.sh`命令。
经过一段时间的训练和测试，可以看到运行结果。
WER为0.00。看来这个例子识别的还是挺准的。
PS:WER（WordError Rate）是字错误率，是一个衡量语音识别系统的准确程度的度量。其计算公式是WER=(I+D+S)/N，其中I代表被插入的单词个数，D代表被删除的单词个数，S代表被替换的单词个数。也就是说把识别出来的结果中，多认的，少认的，认错的全都加起来，除以总单词数。这个数字当然是越低越好。
下面进入./yesno/s5/waves_yesno目录瞧一瞧。
全部都是.wav格式的音频文件。可以打开一个文件听一听，发现是一个老男人连续不停地说yes或者no，每个文件说8次。文件名中，0代表那个位置说的是no，1代表说的是yes。这个实验没有单独的标注文件，直接采用的是文件名来标注的。

## 运行timit实例
Timit是LDC搜集的一个语料库，TIMIT语音库有着准确的音素标注，由630个话者组成，每个人讲10句，美式英语的8种主要方言，是一个学习用的好例子。但是由于这个数据库是商业用的，所以Kaldi里面并没有附带数据。

1. 在timit/s5文件夹下面新建文件夹data；

2. 把timit.rar解压后的四个文件放入data中，包括：DOC, TEST, TRAIN三个文件夹和README.DOC文件；

3. 修改s5文件夹下的cmd.sh，因为是在虚拟机上跑的，所以代码修改为单机版，其它代码都注释掉，只保留下面4行：
```bash
export train_cmd=run.pl
export decode_cmd=run.pl
export cuda_cmd=run.pl
export mkgraph_cmd=run.pl
```

4. 然后path.sh定义各种工具包的路径。path.sh修改如下：
```bash
export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
#export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export IRSTLM=$KALDI_ROOT/tools/irstlm
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
```

5. 修改s5文件夹下的run.sh，修改timit=开头的那行代码即可，该行代码是告诉程序我们下载好的语料库的位置在哪里，例如我修改后的代码为：
```bash
#timit=/export/corpora5/LDC/LDC9351/timit/TIMIT # @JHU
timit=/root/kaldi-trunk/egs/timit/s5/data
```
run.sh中SGMM2 Training & Decoding部分exit 0没有注释掉，导致这里直接停了下来，需要注意检查一下。

6. cd到s5目录下，执行./run.sh

7. 执行第五步的时候很有可能会报错，错误信息与irstlm相关，这是因为这个例程建立语言模型是用irstlm工具建立的，但是在最新版本的kaldi里面，irstlm不是默认编译的。所以我们先得自行编译irstlm。
首先进入kaldi目录下的tools/extras目录，执行`./install_irstlm.sh`脚本。安装完成以后，目录下出现irstlim目录。由于timit例程里面的引用irstlm工具的路径是tools目录，所以把这个目录拷贝到tools/目录下。回到egs/timit/s5目录，执行`./run.sh`脚本，于是timit例程就跑起来了：

8. s5目录下可以打开RESULTS文件，这个文件是最近的运行结果（我没有运行出来，是别人运行好的结果）