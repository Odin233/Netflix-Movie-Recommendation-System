## 目录

  - Python Project
  - 神经网络相关
    - tensorflow调用GPU进行训练的前置步骤-配置CUDA和cuDNN
    - 补充：PyTorch安装（和Tensorflow同级，同样需要先配置CUDA和cuDNN）
    - 检测是否使用GPU运行程序，并过滤tensorflow的警告和错误
    - 使用tensorflow2.0运行tensorflow1.0的语句所需的改动
    - 使用Python2运行tensorflow的注意事项
  - Python程序降低时间和空间复杂度的具体优化方案
    - 转换平台Matlab计算
    - Tensorflow和keras（算法）
      - Embedding层
    - tensorflow优化方法
      - Estimator计算图
      - tf.data
        - tf.data的prefetch
        - tf.data的parallel mapping
      - tensorflow2图执行模式
    - scipy-稀疏矩阵（算法）
    - cupy（调用GPU）
    - cupy和numpy的内存数据异步传输，节省时间和内存空间（异步数据传输）
    - Numba（更高效的编译器）
    - 内存和算力硬件提升（云算力）
      - AutoDL

## Python Project

大二下AI Project，让我真正入门了深度学习，实践了深度学习的一个简单案例，为什么需要调用GPU，如何调用GPU，怎么更好地调用GPU。

VS2019：

- VS2019，解决方案资源管理器，右键-添加-已有项，即可添加已有的.py文件。

Gallery和Probe：

- Training set是与特定用户无关（也就是不包含Gallery set和Test set中的数据）但特征与特定用户的数据具有相似性的数据组成的训练用数据库。
- Gallery set是特定用户的数据组成的测试用数据库。
- Probe set是待验证的特定用户（有可能包含不在Gallery set中的特定用户，模型仍会输出最可能的匹配结果）的数据组成的测试用数据库。相当于Test set。
- 模型通过使用Training set训练得到。对训练好的模型输入一个来自Probe set的数据，模型判断此输入最可能匹配Gallery set中的哪一个用户。

协同过滤（CF）：

- 协同过滤技术可划分为基于内存/邻域（Memory-based）的协同过滤与基于模型的协同过滤技术（Model-based CF）。
- 基于模型的协同过滤技术中矩阵分解（Matrix Factorization，MF）技术最为普遍和流行，因为它的可扩展性好并且易于实现。

基于矩阵分解（MF）的隐语义模型（LFM）概述：

- 基于矩阵分解的推荐算法的核心假设是用隐语义（隐变量）来表达用户和物品，他们的乘积关系就成为了原始的元素。这种假设之所以成立，是因为我们认为实际的交互数据是由一系列的隐变量的影响下产生的（通常隐变量带有统计分布的假设，就是隐变量之间，或者隐变量和显式变量之间的关系，我们往往认为是由某种分布产生的。），这些隐变量代表了用户和物品一部分共有的特征，在物品身上表现为属性特征，在用户身上表现为偏好特征，只不过这些因子并不具有实际意义，也不一定具有非常好的可解释性，每一个维度也没有确定的标签名字，所以才会叫做 “隐变量”。而矩阵分解后得到的两个包含隐变量的小矩阵，一个代表用户的隐含特征，一个代表物品的隐含特征，矩阵的元素值代表着相应用户或物品对各项隐因子的符合程度，有正面的也有负面的。

- 数据处理：行为用户ID，列为电影ID，数据为评分，无视时间影响。
- 将预测值（两通过分解得到的用户-属性矩阵和物品-属性矩阵相乘得到的预测评分矩阵）与样本评分矩阵中已有评分的值构造平方差损失函数。
- 对数据处理得到的矩阵进行矩阵分解（不同算法决定分解方式），$k$为隐变量个数，使平方差损失函数最小化，转化为最优化问题。
- 通过交替最小二乘法（ALS）和随机梯度下降法（或更多方法）均可同时得到两分解得到的矩阵。
- 如果需要得到某一用户/电影的相似用户/电影，只需要对分解得到的两个矩阵取对应的用户/电影的隐变量向量，和其他所有用户/电影取欧式距离，最小前几名即为相似（用户喜好/电影属性相似）。

ALS：

两分解得到的矩阵为P和Q，先固定其中一个矩阵，以另一个作为变量，通过损失函数最小化求出作为变量的矩阵，然后固定和作为变量矩阵交换，不断重复。直到损失函数收敛（损失函数的值满足阈值条件）或到达迭代上限。

随机梯度下降法：

对损失函数的各个参数分别求偏导，得到各个参数的梯度，各个参数分别进行梯度下降（两分解得到的矩阵是同时对所有参数进行更新的），直到到达迭代上限。

数据预处理需要的函数：

```
# 对于一维数组或者列表，np.unique() 函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表。
```

神经网络训练术语：

- epoch：所有的数据送入网络中， 完成了一次前向计算 + 反向传播的过程。当epoch为多个时，每一个epoch相当于在上一个epoch训练结束的位置继续训练，即从上个epoch训练好的参数开始继续训练。
- batchsize：每个batch中训练样本的数量。
- iterations：完成一次epoch需要的batch个数。

假设训练集有1000个样本，设置batchsize=10，那么训练完整个样本集（1次epoch）需要进行100次iteration。

## 神经网络相关

神经网络一共分为三个大类。

图网络：图网络定义在图结构上，节点可以由一个或一组神经元构成，节点之间可以是有向或无向的。其演算方式和前馈神经网络、反馈神经网络均不同。

前馈神经网络：结构分为输入层、隐藏层、输出层。整个网络中的信息是朝着同一个方向传播的，没有反向的信息传播（除了更新权重，神经元中的信息只会向下一层的神经元传递）。包括全连接前馈神经网络、卷积神经网络。

反馈神经网络：结构分为输入层、隐藏层、输出层。反馈神经网络中的神经元除了接收其他神经元的信息，还可以接收自己的信息。反馈神经网络中的信息是可以单向和双向传播（除了更新权重，神经元中的信息可以向上一层的神经元传递）。反馈神经网络中的神经元具有记忆功能。包括循环神经网络（使用外部记忆单位和读写机制，称为记忆增强网络）。

### tensorflow调用GPU进行训练的前置步骤-配置CUDA和cuDNN

Windows下安装CUDA和cuDNN需要手动下载并安装，且不与Anaconda中的虚拟环境共通，适用于整个Windows操作系统的命令行环境（也就是手动下载Python并安装的所处环境）。

Tensorflow安装时会自动安装对应的CUDA版本，但可能出错，此时需要更换tensorflow版本。

[CUDA与cuDNN安装教程（超详细）-CSDN博客.mhtml](/_resources/CUDA与cuDNN安装教程（超详细）-CSDN博客.mhtml)

ctrl+c 强制中断Python程序的运行，这包括经常卡住的pip install函数。

1.查看电脑的CUDA版本：

第一种方法：NVIDIA控制面板-系统信息-组件-NVCUDA64.DLL-产品名称NVIDIA CUDA  XX.XX.XX driver。（XX.XX.XX即为CUDA版本）（查看的是GPU驱动包自带的CUDA版本，即CUDA Driver Version）

第二种方法：命令行窗口输入nvidia-smi，可以同时查看Driver Version和CUDA Version。调用的是CUDA的driver API。（查看的是GPU驱动包自带的CUDA版本，即CUDA Driver Version）

第三种方法：命令行窗口输入nvcc -V，可以查看CUDA Version。（如果与第一和第二种方法查看得到的版本不同，则实际使用中以第三种方法为准，因为第一和第二种方法查看得到的是当前显卡驱动版本支持的最大CUDA toolkit版本，而不是实际安装的CUDA toolkit版本，实际运行的是实际安装的CUDA toolkit版本，安装的其他依赖适配的版本也是看实际安装的CUDA toolkit版本）调用的是CUDA的runtime API。（查看的是CUDA Toolkit的CUDA版本，即CUDA Runtime Version，如无安装CUDA Toolkit则无法查看）

2.然后查看显卡驱动版本和CUDA Toolkit版本的对应关系：

- 命令行窗口输入nvidia-smi，获得显卡驱动版本和对应的CUDA Toolkit可支持的最新版本：

![bec9f68f3535d2d646d50d9e9b202574.png](/_resources/bec9f68f3535d2d646d50d9e9b202574.png)

- 在以下网址也可以查询显卡驱动版本和对应的CUDA Toolkit可支持的最新版本：

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

![947ee69242a88af24966b1fa82948b37.png](/_resources/947ee69242a88af24966b1fa82948b37.png)

- python、tensorflow_gpu和cuDNN、CUDA Toolkit的版本部分对应图：

![08a76d656a7dffc685c4f7b780491e46.png](/_resources/08a76d656a7dffc685c4f7b780491e46.png)

3.在控制面板卸载本来的CUDA：

除了NVIDIA的图形驱动程序、NVIDIA Physx系统软件、NVIDIA GeForce Experience以外的全部卸载掉。（"以外的"不是指全部带有"NVIDIA"的都卸载，而是带有"NVIDIA CUDA XXX"和"NVIDIA Frameview"的，不确定也没关系，在安装CUDA Toolkit时会提示安装失败的原因指出哪些需要卸载）

4.下载CUDA Toolkit（版本一般不选显卡驱动版本的可支持的最新版本，而是稍旧一点，除非有特定需求）：

https://developer.nvidia.com/cuda-toolkit-archive

安装过程略，和普通的安装包一样，路径建议不要更改。和VS2019有关的那一项建议不勾选。

安装完成后，命令行窗口输入nvcc -V来验证是否安装成功。若提示没有有此命令，则去添加环境变量后再次尝试。

5.下载cuDNN（官网注册太过繁琐，建议直接使用迅雷链接下载）：

https://developer.nvidia.com/rdp/cudnn-archive

把下载得到的压缩包解压后的所有子文件夹复制到CUDA的安装根目录即配置好cuDNN。

默认路径：C:\ProgramData\NVIDIA GPU Computing Toolkit\v版本号\

然后在系统环境变量的Path中新增CUDA的安装目录：

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v版本号

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v版本号\lib\x64

6.安装tensorflow-gpu：

激活anaconda虚拟环境后使用pip安装：

```
pip install --ignore-installed --upgrade tensorflow-gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在此之前需要先安装普通的tensorflow。普通的tensorflow版本仅需要和python版本对应即可，不需要和tensorflow-gpu的版本对应。

7.在IDE中检验是否调用GPU进行训练：

```py
import tensorflow as tf
print(tf.test.is_gpu_available())
# 返回True则为成功调用GPU。
```

8.如果要使用GPU进行训练，则需要将 cudnn64_8.dll文件手动放入C:\Windows\System路径下。（按上述步骤安装好后应该在cuDNN的bin文件夹里，但是直接运行程序找不到）

### 补充：PyTorch安装（和Tensorflow同级，同样需要先配置CUDA和cuDNN）

PyTorch安装时会自动安装对应的CUDA版本，这在虚拟环境中配置非常方便，但是cuDNN不会自动安装，手动使用`conda install cudnn`安装（自动匹配CUDA版本）。

根据CUDA版本搜索对应的PyTorch版本的安装指令（包含了PyTorch本体和其他相关的PyTorch组件）：

https://pytorch.org/get-started/previous-versions/

CUDA版本为11.2是特殊情况（视为11.1版本）：

`pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host pypi.org --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org`

PyTorch本体安装包太大，若网速太慢，复制下载时显示的URL，打开浏览器输入URL用vpn下载，再关闭vpn在Anaconda本地安装，注意Windows下Anaconda的安装均需要关闭vpn，即使是本地安装也需要联网（本地安装需要使用可切换不同磁盘路径的Anaconda Powershell Prompt来执行安装指令而不是Anaconda Prompt）：

1.9.1版本PyTorch本体下载URL：https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp38-cp38-win_amd64.whl

[anaconda中pip方式离线安装pytorch_linux anaconda3 pip 离线安装库-CSDN博客.mhtml](/_resources/anaconda中pip方式离线安装pytorch_linux%20anaconda3%20pip%20离线安装.mhtml)

`pip install C:\文件名`

若出现错误：`ERROR: xxx is not a supported wheel on this platform.`原因是文件名格式不合法，使用`pip debug --verbose`查询合法的文件名格式，来修改文件名。

![3b632005e83b0e88b4f5fb1bfd14dde9.png](/_resources/3b632005e83b0e88b4f5fb1bfd14dde9.png)

其他PyTorch组件很小，可以直接用pip安装：

`pip install torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html --trusted-host pypi.org --trusted-host download.pytorch.org --trusted-host files.pythonhosted.org`

### 检测是否使用GPU运行程序，并过滤tensorflow的警告和错误

检测当前使用的设备类型：

```py
from tensorflow.python.client import device_lib
# 导入检测设备的函数。
 
print(device_lib.list_local_devices())
# 输出此时使用的设备类型和参数。（代码位置会影响结果，一开始默认使用CPU，检测时应当放在调用GPU的代码的后面）
```

过滤一部分错误和警告：

```py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 3代表的是只有代码出错了才会显示，否则不显示。
# 2和1,2代表输出错误和警告。
# 1代表默认设置，显示所有信息。
```

设置tensorflow日志输出级别（也能过滤一部分错误和警告）：

```py
import warnings
warnings.filterwarnings('ignore') 
# 设置tensorflow日志输出级别。
# 放的位置也会影响效果。
```

### 使用tensorflow2.0运行tensorflow1.0的语句所需的改动

导入库部分：

```py
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# 因为使用的库是tensorflow2，因此用以上两句替换import tensorflow as tf，使得语法和函数符合tensorflow1的规范。
```

### 使用Python2运行tensorflow的注意事项

如果要在python2的py文件里面写中文，则必须要添加一行声明文件编码的注释，否则python2会默认使用ASCII编码。

```py
# -*- coding:utf-8 -*-
```

## Python程序降低时间和空间复杂度的具体优化方案

小数据量或简单模型可以通过numpy等（方便）使用CPU或GPU进行训练。大数据量或复杂模型必须通过深度学习框架（学习成本高）使用GPU进行训练。定位不同。

- 减少非必要计算（根据算法原理可省略的一些影响不大的计算）和内存占用（不赋值中间变量，直接计算最终结果赋值给最终变量，以免内存不释放）。
- 优化算法，尽量避免算法瓶颈。
- 优化算法和参数设置，使得在合适范围内尽量提高CPU、GPU、内存的利用率（充分利用，避免过低）。

### 转换平台Matlab计算

### Tensorflow和keras（算法）

CUDA支持数据拷贝与数值计算并存。使用Tensorflow需要有对应版本的CUDA。

![7807eb981c49c4e28201463b7e249248.png](/_resources/7807eb981c49c4e28201463b7e249248.png)

深度学习的数据加载流程：

CPU内存大，因为除了RAM，还能将硬盘的一部分空间充当虚拟内存。GPU内存（显存）小，除了本身的显存，只能通过GPU共享内存（一部分且空闲的RAM）进行拓展，无法调用硬盘充当的虚拟内存。因此，目前的主流深度学习框架，均会实现数据从硬盘读取到CPU内存，然后CPU将其传输到GPU内存，GPU完成计算后，将结果传输回CPU内存。

先将硬盘中的数据读取到内存中，然后CPU从内存中读取数据，再将数据传输到显存中，等待GPU处理，处理完成之后会传递一部分参数数据到CPU，CPU将这些参数数据传输到内存中。

深度学习框架的优点：

- 默认调用GPU进行并行计算。
- 业内顶级的优化，已有算法的时间复杂度和空间复杂度已被优化过。几乎不需要担心算法的瓶颈。
- 能够充分调用GPU和CPU的性能。准确来说，深度学习框架已经替用户实现了在cpu处理数据，并且分成多个batch，从cpu内存传输到gpu内存（显存），完成计算后传输回cpu内存中，整个过程。实现cpu承担主要空间负载，gpu承担主要计算负载。
- 支持将数据集分为多个batch（batch大小根据显存来自己设置）依次训练，防止显存不足。不需要担心数据集大小的限制。
- 当专用GPU内存（显存）不够时，深度学习框架会自动使用共享GPU内存。
- 一次性实现了深度学习中的诸多功能并进行了业内顶级的优化，深度学习框架的源码值得学习。

```
nvidia-smi -h
% cmd内执行。查看参数。
nvidia-smi -l 0.5
% 每隔0.5秒，打印NVIDIA显卡各项参数，主要是利用率、显存、功率。

import tensorflow as tf
print(tf.test.is_gpu_available())
# True表示GPU可用。

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 指定使用GPU0和GPU1（多GPU并行，分布式训练）。
# 在只有一张显卡的机器上不需要此代码指定gpu，多显卡机器才需要。
```

#### Embedding层

![f08154e0ade74d5775333a0f704c76c1.png](/_resources/f08154e0ade74d5775333a0f704c76c1.png)

```py
tensorflow.keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

- 例如，代表全部种类word的ID为0-99，ID数字连续（一共有100种word，则ID为0-99，不会出现不连续的情况），排列不连续（可能数据顺序不按照word的ID连续，可能每种word有多个数据），则嵌入层的参数input_dim=99+1=100，代表输入嵌入层的word将会转化为最多100个one-hot编码进行表示（通过0和1排列，每个独特的排列代表一种word）。

- 如果不慎使得input_dim设置得比实际word种类数更多进行训练，其实没有影响。因为多出来的ID对应的word的数据不存在，因此不影响两个分解矩阵（embedding层权重矩阵）和最终评分矩阵（两分解矩阵相乘）的已有数据部分，直接去掉多出来的行列即可（也可以保留，为以后可能的更多种类的word训练做准备）。

- embedding层将输入的word转化为一个k维的向量（不是矩阵），word之间的相似性可以通过它们的k维向量的特征（例如向量之间的欧式距离等）反映出来，这是因为训练过程中越相似的word更新的次数和方式越趋于相同。

- output_dim其实就是k的值，代表word转换为向量的维度数。input_length为可选参数，为输入序列的长度，也就是每个样本（句子）需要用多少个word进行表示（一般取最长的样本的word数，因为其需要最多的word个数进行表示，其他更短的样本则需要在对应位置补0，因此对word转化为one-hot编码时会编码为除0外的整数，此整数即为0和1排列表示的二进制数对应的十进制整数）。batch_size即为输入的样本数。

- 因此embedding层的输入维度为(batch_size,input_length)，输出维度为(batch_size,input_length,output_dim)。

- 在矩阵分解中，根据神经网络的原理，对于两种输入各自更新两个嵌入层的权重，最后提取两个嵌入层的权重恰好就是两个分解得到的矩阵。

### tensorflow优化方法

降低CPU时间，提高GPU时间，提高GPU利用率。

#### Estimator计算图

一般训练的经典结构：

```
create_model()
2. create_model_saver()
3. create_summary_writer()
4. create_session()
5. do_init()
6. for i in range(num_train_steps):
7.      load_batch(...)                # cpu
8.      preprocess(...)                # cpu
9.      feed_dict = {...}              # cpu
10.     fetch_list = [...]             # cpu
11.     buf = session.run(fetch_list, feed_dict)    # gpu
12.     postprocess(buf)               # cpu
13.     print(...)                     # cpu
14.     if i % x == 0:
15.         summary_writer.write(...)  # cpu
16.     if i % xx == 0:
17.         model_saver.save(...)      # cpu
```

tf.estimator将上述的各项操作进行了封装（计算图），提高CPU加载数据等操作的效率。tf.estimator把大部分的代码写进计算图里了，但是从数据的载入和预处理依然是在CPU里串行进行，因此使得CPU对数据的载入和预处理并行进行，就是最后的瓶颈。

#### tf.data

很天然地支持以streaming的方式读取数据，适用于大数据集。

不用tf.data：

![3901e5c158770ba3dd27213361bbf441.png](/_resources/3901e5c158770ba3dd27213361bbf441.png)

使用tf.data的目标：

![48341830deab2da1095020ff97be6ecf.png](/_resources/48341830deab2da1095020ff97be6ecf.png)

##### tf.data的prefetch

预先获取下一个step要load的batch。对tf.data.Dataset的prefetch()函数的buffer_size参数进行赋值，比如buffer_size=1（一般设置为`tf.data.experimental.AUTOTUNE`，自动选择合适的数值），那么模型每次prepare完一个batch后，就会自动再额外的prepare一个batch，这样下一个train step到来的时候就可以直接从内存中取走这个事先prepare好的batch。

如果prepare一个batch耗时很短的话确实两全齐美，但是如果耗时比较久，尤其一下子prefetch好几个batch的话，一旦prepare的用时超过了train一个step的用时，那么每个train step的性能就会受限于prepare的效率。如图所示：

![d6089a74cf229283ccab8ca172814586.png](/_resources/d6089a74cf229283ccab8ca172814586.png)

为确保prepare阶段的用时小于train阶段的用时，使用parallel mapping。

##### tf.data的parallel mapping

让CPU多线程并行，提高加载数据等操作的速度。对tf.data.Dataset的map()函数的num_parallel_calls参数（一般设置为`tf.data.experimental.AUTOTUNE`，自动选择合适的数值）进行赋值：

![a5a15766370df7b80d453bc431b8c8a8.png](/_resources/a5a15766370df7b80d453bc431b8c8a8.png)

这样的话只要prefetch的buffer_size和map的num_parrellel_calls取得合适，基本就可以实现不间断的train啦，也就是几乎达到100%的GPU利用率！

#### tensorflow2图执行模式

函数前加上修饰符：`@tf.function`。注意点：

- 函数内尽量使用tf原生函数（也就是尽量将函数使用tf的API来写），如`print`改为`tf.print`，`range`改为`tf.range`。
- 避免在`@tf.function`修饰的函数内部定义`tf.Variable`。
- 注意，被`@tf.function`修饰的函数，不可修改该函数外部的列表和字典等包含数据结构的变量。

### scipy-稀疏矩阵（算法）

```
import numpy as np
from scipy.sparse import csc_matrix,csr_matrix,find
import timeit

A=np.array([[1,2,3],[4,5,6],[7,8,9]])
print("原矩阵：")
print(A)
print(type(A))

print("稀疏矩阵：")
sA= csc_matrix(A)
print(sA)
print(type(sA))

print("稀疏矩阵转化为原矩阵：")
nA=sA.toarray()
print(nA)
print(type(nA))

print("稠密矩阵：")
mA=sA.todense()
print(mA)
print(type(mA))

# 将matrix类型转换为ndarray类型
print("稠密矩阵（ndarray类型）：")
nA=mA.A
nA=np.asarray(mA)
print(nA)
print(type(nA))

row = np.array([0,2,2,0,1,2])
col = np.array([0,0,1,2,2,2])
data1 = np.array([1,2,3,4,5,6])
data2 = np.array([10,20,30,40,50,60])
X = csc_matrix((data1, (row, col)), shape=(3, 3))
Y = csc_matrix((data2, (row, col)), shape=(3, 3))

print("X，原矩阵")
print(X.toarray())

print("Y，原矩阵：")
print(Y.toarray())

print("X，返回非零元素的位置和值，可保存下来用于构建稀疏矩阵，节省内存")
row = find(X)[0]
col = find(X)[1]
data = find(X)[2]
print(row)
print(col)
print(data)

print("X.T，原矩阵：")
print(X.T.toarray())

print('X**2，稀疏矩阵中对应元素的2次方，原矩阵')
print((csc_matrix.power(X,2)).toarray())

print('X*Y，稀疏矩阵中矩阵的乘法，原矩阵')
print((X*Y).toarray())

print('X*Y，稀疏矩阵中矩阵的乘法，原矩阵')
print((csc_matrix.dot(X,Y)).toarray())

print('X*Y，稀疏矩阵中对应元素的乘法，原矩阵')
print((csc_matrix.multiply(X,Y).toarray()))

print('X+Y，稀疏矩阵中对应元素的加法，原矩阵')
print((X+Y).toarray())

print('X，计算出来的结果是exp(x)-1(e^{x}-1),稀疏矩阵中对应元素的计算')
print(( csc_matrix.expm1(X)).toarray())

X=np.random.uniform(size=(2000,2000))
Y=np.random.uniform(size=(2000,2000))
Z=np.random.uniform(size=(2000,2000))
X[X<0.7]=0
Y[Y<0.7]=0
Z[Z<0.7]=0
x=X
y=Y
X=csc_matrix(X)
Y=csc_matrix(Y)
Z=csc_matrix(Z)

print("原矩阵Z大小：{}".format(Z.toarray().nbytes))
print("稀疏矩阵Z大小：{}".format(Z.data.nbytes+Z.indptr.nbytes+Z.indices.nbytes))

def numpy_dot(x,y):
  return np.dot(x,y)

print("numpy矩阵乘法时间：")
print(timeit.timeit(lambda:numpy_dot(x,y), number=1, globals=globals()))

def sparse_dot(X,Y):
  return X.dot(Y)

print("scipy稀疏矩阵乘法时间：")
print(timeit.timeit(lambda:sparse_dot(X,Y), number=1, globals=globals()))
```

关于numpy矩阵和scipy矩阵，稠密矩阵和稀疏矩阵：

- 使用scipy.sparse稀疏矩阵来降低进行矩阵运算时的空间和时间复杂度，但是无法利用cupy了。因为numpy.dot()函数不支持稀疏矩阵的运算。

关于对numpy库和scipy库，调用GPU并行计算：

- 通过CUDA和numba。

### cupy（调用GPU）

使用pip安装与CUDA版本相匹配（通过nvcc -V指令查看得到的版本）的cupy库，其为numpy库的镜像，函数基本一一对应。cupy调用CUDA使用GPU进行计算，擅长矩阵计算。

缺点：相比于CPU可以调用内存和虚拟内存（系统设置可以任意分配空闲硬盘空间为虚拟内存），GPU只能使用显存，而显卡的显存一般不大。也就是大数据量时，只能降低精度（单精度和半精度），或者把任务分成几部分，或者通过算法使得将大数据矩阵使用numpy存储（CPU计算，内存，虚拟内存）而将仅涉及计算的子数据矩阵使用cupy存储（GPU计算，显存）和计算，来避免显存不够用。这是cupy和深度学习框架相比的缺点，但是cupy对代码的适用性更广。

### cupy和numpy的内存数据异步传输，节省时间和内存空间（异步数据传输）

### Numba（更高效的编译器）

`pip install numba==0.52.0`是兼容tensorflow(2.5.0)-Python(3.8)-CUDA(11.2)的版本。

numba将python原来的CPython编译器更换为能够编译为等效机器代码的编译器，仅限数组运算，擅长处理for循环和numpy函数。

numba想要适用越复杂的代码，需要的代码改写越繁杂，因此对代码污染性较强。

### 内存和算力硬件提升（云算力）

云算力GPU租赁。

#### AutoDL

选择合适的GPU-CPU（自动）-内存（自动）-数据盘（自动），深度学习框架（例如tensorflow），租用一个实例/容器，其实可以说是一台虚拟机（因此也有开关机）。一般选择按小时计费（开机就算）。

将需要用到的数据上传（网盘，lab页面上传，ftp上传），代码上传，就可以当作自己电脑一样跑程序了。但注意为Linux系统的服务器。

帮助文档：

https://www.autodl.com/docs/

注意事项：

- 在AutoDL网页的终端键入命令运行时，通常是一定要先cd进入到autodl-fs（网盘）目录下（相对路径命令是cd autodl-fs），不然可能会命令失败，找不到路径（哪怕是绝对路径也可能会有bug）

- 绝对路径前面的最前面是/root/

- Linux终端中，空格使用`\ `来表示。

- cd指令分为相对路径和绝对路径。

- 只要jupyterlab不出现重启（几乎不会），jupyterlab的终端就会一直运行，无论是本地主机断网还是关机。如果关机了这个终端tab，可以在Jupyter Lab的左侧栏正在运行终端和内核中找回。

- ssh指令是指用于另一台设备登录当前实例的Linux指令。因此，两台设备传输文件，需要被传输的设备开机（登录指令为：`ssh -p 端口号 IP地址`，例如`ssh -p 12345 region-3.autodl.com`，但是不需要在终端运行），需要传输的设备终端输入`scp -rp 端口号 传输的文件在传输的设备上的路径 IP地址:传输的文件在被传输设备上的路径(注意此路径不能用\ 而是\\ 来表示空格，需要使用""括住整个路径)`（例如`scp -rP 12345 /root/autodl-fs/xxx region-3.autodl.com:/root/autodl-fs/`），登入root账号后就会直接开始文件传输。

- 最可靠的还是通过网盘一个一个文件地进行传输，scp指令传输经常会断开，且无法找回正在运行的终端。

- 出现显卡利用率低的情况，需要具体分析瓶颈。可能是深度学习框架和CUDA版本没对上。可能是代码没有使用GPU。可能是日志IO操作频率过高（训练一瞬间，写日志一会，导致GPU利用率低）。可能是使用的数据集的体量不同（CPU加载数据，从内存传输到显存等操作需要的时间更长，GPU瞬间完成计算任务后闲置，造成GPU利用率低，但是GPU内存占用高）。如果处理的数据很多很大，那么大部分显存可能用来存储读写动量、梯度之类的，造成显存明明占满了，但实际利用率很低的情况。

- 先拿子集或同类型小型数据集来做调试和测试。对于数据量大的模型训练，你不会希望等待一万年来观察结果来调试代码，你也不会希望训练过程中途因为一些奇怪的原因报错中断而不自知。

- 最好以tf格式保存模型。如果您自定义了任何custom_loss（自定义损失函数）。h5格式将不会保存优化器状态，无法在保存的模型基础上继续训练模型。

深度学习框架下GPU利用率优化方法：

- 算法的优化。

- 结合深度学习框架的数据加载流程，对CPU和GPU合理分配任务。

- 提高数据预处理效率（并行）。增加CPU线程数（并行加载数据），tensorflow通过workers参数设置，Pytorch设置num_workers设置，取值一般为2、4、6、8、16。数据量越大效果越显著。

- 使用pin_memory技术。可以减少数据传递到显存中的时间。

- 增大batch_size（减小iteration），尽量刚好占满GPU内存（注意tensorflow和keras会自动占满显存，此时只能通过观察任务管理器的共享GPU内存来得知实际是否占满显存），使得GPU的计算任务更多，避免出现CPU疯狂加载数据，而GPU完成计算任务后处于空闲状态（利用率极低）的情况。对batch_size进行设置会影响一次epoch所需要的迭代次数（数据加载增多的所需时间可以忽略不计），影响内存和显存利用率以及大矩阵乘法的并行化效率，影响模型的收敛速度和收敛精度。一般在batch_size增加的同时，我们需要对epoch进行增加，以达到以更小的batch_size训练更少epoch相同的收敛精度（batch_size的增大会使得每个epoch的loss下降更慢）。在模型结构固定的情况下，尽量将batch size设置大，充分利用GPU的内存。

综上所述：

- GPU会很快的算完从CPU内存传输来数据。
- GPU不适合计算float64，效率很低。一般为float32和float16。
- 模型的大小和训练时选择得到batch size影响显存占用率。模型的种类和模型的训练算法对并行计算的适应性影响GPU利用率。
- Tesnsorflow深度学习框架默认将GPU显存全部空占，无法得到真正的显存利用率。
- 可能是CPU性能不足，数据处理算法优化不好。CPU完成数据处理所需时间长，会造成GPU空闲时间多，GPU利用率低。可以通过不输出日志和不记录中间数据减少CPU任务量。（load下一个batch、预处理这个batch以及在GPU上跑出结果后打印日志、后处理、写summary甚至保存模型等，这一系列的花销都要靠CPU去完成）
- 可能是没有根据任务量和性能设置CPU和GPU同步工作的逻辑和任务分配，CPU和GPU无法同时工作，GPU利用率低。
- 可能是batch size设置不合理，导致GPU显存占用不高，计算任务轻，空闲时间多，GPU利用率低。
- 可能是内存带宽不足、CPU的IO传输吞吐量达到了瓶颈，与batch size设置不合理的表现类似（一个是传输速度不够，一个是设置传输量太少，两者是不同的，但是造成的问题是类似的），表现为GPU显存占用率低，GPU利用率低。