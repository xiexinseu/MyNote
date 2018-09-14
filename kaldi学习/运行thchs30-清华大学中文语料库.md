# 运行thchs30-清华大学中文语料库

[TOC]



Kaldi中文语音识别公共数据集一共有4个（据我所知），分别是：


1. aishell: AI SHELL公司开源178小时中文语音语料及基本训练脚本，见kaldi-master/egs/aishell

2. gale_mandarin: 中文新闻广播数据集(LDC2013S08, LDC2013S08）

3. hkust: 中文电话数据集(LDC2005S15, LDC2005T32)

4. thchs30: 清华大学30小时的数据集，可以在http://www.openslr.org/18/下载

今天我们来运行thchs30数据集。

## 数据准备
首先我们需要下载语料库：

下载地址为：http://www.openslr.org/18/
里面共有3个文件，分别是：
data_thchs30.tgz [6.4G] ( speech data and transcripts )
test-noise.tgz [1.9G] ( standard 0db noisy test data ) 
resource.tgz [24M] ( supplementary resources, incl. lexicon for training data, noise samples )
下载后随便解压到一个文件夹里，例如在egs/thchs30/s5下新建了一个文件夹thchs30-openslr，然后把三个文件解压在了该文件夹下
这个数据集包含以下内容：

下载后随便解压到一个文件夹里，例如在egs/thchs30/s5下新建了一个文件夹thchs30-openslr，然后把三个文件解压在了该文件夹下

这个数据集包含以下内容：


| 数据集  |   音频时长(h) |   句子数  |   词数    |
| - | - | - | - |
| train(训练) |   25      |  10000    |   198252   |
| dev(开发)   |   2:14    |   893     |   17743   |
| test(测试)  |   6:15    |   2495    |   49085   |

还有训练好的语言模型word.3gram.lm和phone.3gram.lm以及相应的词典lexicon.txt。

其中dev的作用是在某些步骤与train进行交叉验证的，如local/nnet/run_dnn.sh同时用到exp/tri4b_ali和exp/tri4b_ali_cv。训练和测试的目标数据也分为两类：word（词）和phone（音素）。

1.local/thchs-30_data_prep.sh主要工作是从`$thchs/data_thchs30`（下载的数据）三部分分别生成word.txt（词序列），phone.txt（音素序列），text（与word.txt相同），wav.scp（语音），utt2pk（句子与说话人的映射），spk2utt（说话人与句子的映射）

2.#produce MFCC features是提取MFCC特征，分为两步，先通过steps/make_mfcc.sh提取MFCC特征，再通过steps/compute_cmvn_stats.sh计算倒谱均值和方差归一化。

3.#prepare language stuff是构建一个包含训练和解码用到的词的词典。而语言模型已经由王东老师处理好了，如果不打算改语言模型，这段代码也不需要修改。
a)基于词的语言模型包含48k基于三元词的词，从gigaword语料库中随机选择文本信息进行训练得到，训练文本包含772000个句子，总计1800万词，1.15亿汉字
b)基于音素的语言模型包含218个基于三元音的中文声调，从只有200万字的样本训练得到，之所以选择这么小的样本是因为在模型中尽可能少地保留语言信息，可以使得到的性能更直接地反映声学模型的质量。
c)这两个语言模型都是由SRILM工具训练得到。

## 修改脚本
1.首先修改s5下面的cmd.sh脚本，把原脚本注释掉，修改为本地运行：
```bash
#export train_cmd=queue.pl
#export decode_cmd="queue.pl --mem 4G"
#export mkgraph_cmd="queue.pl --mem 8G"
#export cuda_cmd="queue.pl --gpu 1"
export train_cmd=run.pl
export decode_cmd="run.pl --mem 4G"
export mkgraph_cmd="run.pl --mem 8G"
export cuda_cmd="run.pl --gpu 1"
```
2.然后修改s5下面的run.sh脚本，需要修改两个地方：

第一个地方是修改并行任务的数量，可以根据cpu的个数来定
```bash
#n=4      #parallel jobs
n=2      #parallel jobs
```
第二个地方是修改数据集放的位置，例如我修改的为：
```bash
#thchs=/nfs/public/materials/data/thchs30-openslr
thchs=/media/xiexin/98BC1B8BBC1B62D4/work/ProFromGitHub/kaldi/egs/thchs30/thchs30-openslr
```

## 运行
在s5下执行./run.sh，就会开始运行。
大概有几个过程：数据准备，monophone单音素训练， tri1三因素训练， trib2进行lda_mllt特征变换，trib3进行sat自然语言适应，trib4做quick（这个我也不懂），后面就是dnn了
因为我是在虚拟机上运行的，所以运行的非常慢，建议大家尽量不要再虚拟机上运行，找一个配置好的机器，最好有GPU，这样运行速度比较快。
当运行到dnn时候会报错，因为默认dnn都是用GPU来跑的。它会检查一下，发现只在CPU下，就终止了。这里建议不要跑dnn了，想跑dnn的话还是找GPU吧。

## 算法流程
1.首先用标准的13维MFCC加上一阶和二阶导数训练单音素GMM系统，采用倒谱均值归一化（CMN）来降低通道效应。然后基于具有由LDA和MLLT变换的特征的单音系统构造三音GMM系统，最后的GMM系统用于为随后的DNN训练生成状态对齐。

2.基于GMM系统提供的对齐来训练DNN系统，特征是40维FBank，并且相邻的帧由11帧窗口（每侧5个窗口）连接。连接的特征被LDA转换，其中维度降低到200。然后应用全局均值和方差归一化以获得DNN输入。DNN架构由4个隐藏层组成，每个层由1200个单元组成，输出层由3386个单元组成。 基线DNN模型用交叉熵的标准训练。 使用随机梯度下降（SGD）算法来执行优化。 将迷你批量大小设定为256，初始学习率设定为0.008。

3.被噪声干扰的语音可以使用基于深度自动编码器（DAE）的噪声消除方法。DAE是自动编码器（AE）的一种特殊实现，通过在模型训练中对输入特征引入随机破坏。已经表明，该模型学习低维度特征的能力非常强大，并且可以用于恢复被噪声破坏的信号。在实践中，DAE被用作前端管道的特定组件。输入是11维Fbank特征（在均值归一化之后），输出是对应于中心帧的噪声消除特征。然后对输出进行LDA变换，提取全局标准化的常规Fbank特征，然后送到DNN声学模型（用纯净语音进行训练）。

## 训练与解码脚本解读
本节结合官方文档对主要脚本进行解读。 
以下流程中的符号解释：->表示下一步，{}表示循环，[]表示括号内每一个都要进行一次，()表示不同分支下可能进行的操作 


1.train_mono.sh 用来训练单音子隐马尔科夫模型，一共进行40次迭代，每两次迭代进行一次对齐操作
```bash
gmm-init-mono->compile-train-graphs->align-equal-compiled->gmm-est->
{gmm-align-compiled->gmm-acc-stats-ali->gmm-est}40->analyze_alignments.sh
```
2.train_deltas.sh 用来训练与上下文相关的三音子模型
```bash
check_phones_compatible.sh->acc-tree-stats->sum-tree-stats->cluster-phones->compile-questions->
build-tree->gmm-init-model->gmm-mixup->convert-ali->compile-train-graphs->
{gmm-align-compiled->gmm-acc-stats-ali->gmm-est}35->analyze_alignments.sh
```
3.train_lda_mllt.sh 用来进行线性判别分析和最大似然线性转换
```bash
check_phones_compatible.sh->split_data.sh->ali-to-post->est-lda->acc-tree-stats->sum-tree-stats->
cluster-phones->compile-questions->build-tree->gmm-init-model->convert-ali->compile-train-graphs->
{gmm-align-compiled->gmm-acc-stats-ali->gmm-est}35->analyze_alignments.sh
```
4.train_sat.sh 用来训练发音人自适应，基于特征空间最大似然线性回归
```bash
check_phones_compatible.sh->ali-to-post->acc-tree-stats->sum-tree-stats->cluster-phones->compile-questions->
build-tree->gmm-init-model->gmm-mixup->convert-ali->compile-train-graphs->
{gmm-align-compiled->(ali-to-post->)gmm-acc-stats-ali->gmm-est}35->ali-to-post->
gmm-est->analyze_alignments.sh
```
5.train_quick.sh 用来在现有特征上训练模型。
对于当前模型中在树构建之后的每个状态，它基于树统计中的计数的重叠判断的相似性来选择旧模型中最接近的状态。
```bash
check_phones_compatible.sh->ali-to-post->est-lda->acc-tree-stats->sum-tree-stats->
cluster-phones->compile-questions->build-tree->gmm-init-model->convert-ali->compile-train-graphs->
{gmm-align-compiled->gmm-acc-stats-ali->gmm-est}20->analyze_alignments.sh
```
6.run_dnn.sh 用来训练DNN，包括xent和MPE，
```bash
{make_fbank.sh->compute_cmvn_stats.sh}[train,dev,test]->train.sh->{decode.sh}[phone,word]->
align.sh->make_denlats.sh->train_mpe.sh->{{decode.sh}[phone,word]}3
```
7.train_mpe.sh 用来训练dnn的序列辨别MEP/sMBR。
这个阶段训练神经网络以联合优化整个句子，这比帧级训练更接近于一般ASR目标。 
sMBR的目的是最大化从参考转录对齐导出的状态标签的期望正确率，而使用网格框架来表示竞争假设。 
训练使用每句迭代的随机梯度下降法。 
首先使用固定的低学习率1e-5（sigmoids）运行3-5轮。 
在第一轮迭代后重新生成词图，我们观察到快速收敛。 
我们支持MMI, BMMI, MPE 和sMBR训练。所有的技术在Switchboard 100h集上是相同的，仅仅在sMBR好一点点。 
在sMBR优化中，我们在计算近似正确率的时候忽略了静音帧。
```bash
{nnet-train-mpe-sequential}3->make_priors.sh
```

8.train_dae.sh 用来实验基于dae的去噪效果

```bash
compute_cmvn_stats.sh->{add-noise-mod.py->make_fbank.sh->compute_cmvn_stats.sh}[train,dev,test]->
train.sh->nnet-concat->{{decode.sh}[phone,word]}[train,dev,test]
```

9.train.sh 用来训练深度神经网络模型，帧交叉熵训练，该相位训练将帧分类为三音状态的DNN。这是通过小批量随机梯度下降完成的。
默认使用Sigmoid隐藏单元，Softmax输出单元和完全连接的AffineTransform层，学习率是0.008，小批量的大小为256。 
我们没有使用动量或正则化（注：最佳学习率和隐藏单元的类型不同，sigmoid的值为0.008,tanh为0.00001。 
通过‘–feature-transform’和‘-dbn’将input——transform和预训练的DBN传入此脚本，只有输出层被随机初始化。 
我们使用提前停止来防止过度拟合，为此我们测量交叉验证集合（即保持集合）上的目标函数， 
因此需要两对特征对齐dir来执行监督训练

```bash
feat-to-dim->nnet-initialize->compute-cmvn-stats->nnet-forward->nnet-concat->cmvn-to-nnet->
feat-to-dim->apply-cmvn->nnet-forward->nnet-initialize->train_scheduler.sh
```

10.train_scheduler.sh 典型的情况就是，train_scheduler.sh被train.sh调用。
一开始需要在交叉验证集上运行，主函数需要根据`$iter`来控制迭代次数和学习率。 
学习率会随着目标函数相对性的提高而变化： 
如果提高大于’start_halving_impr=0.01’，初始化学习率保持常数 
否则学习率在每次迭代中乘以’halving_factor=0.5’来缩小 
最后，如果提高小于’end_halving_impr=0.001’，训练终止。


11.mkgraph.sh 用来建立一个完全的识别网络 


12.decode.sh 用来解码并生成词错率结果 


13.align_si.sh 对制定的数据进行对齐，作为新模型的输入 


14.make_fmllr_feats.sh 用来保存FMLLR特征 


15.pretrain_dbn.sh 深度神经网络预训练脚本 


16.decode_fmllr.sh 对发音人自适应的模型进行解码操作 


17.nnet-train-frmshuff.cc 最普遍使用的神经网络训练工具，执行一次迭代训练。过程： 
–feature-transform 即时特征扩展 
NN输入-目标对的每帧重排 
小批量随机梯度下降（SGD）训练 
支持的每帧目标函数（选项 - 对象函数）： 
Xent：每帧交叉熵 
Mse：每帧均方误差 


18.nnet-forward.cc 通过神经网络转发数据，默认使用CPU。选项：
–apply-log :产生神经网络的对数输出(比如：得到对数后验概率) 
–no-softmax :从模型中去掉soft-max层 
—class-frame-counts：从声学得分中减去计算对数的计数

## 专有缩写中文解释
cmvn：倒谱均值和方差归一化

fft：快速傅里叶变换

GMM：高斯混合模型

MFCC：梅尔倒谱系数

pcm：脉冲编码调制

pdf：概率分布函数

PLP：感知线性预测系数

SGMM：子空间高斯混合模型

UBM：通用背景模型

VTLN：特征级声道长度归一化