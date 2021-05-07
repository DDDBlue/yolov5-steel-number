# 声明

1、数据来源：[智能盘点—钢筋数量AI识别](智能盘点-钢筋数量AI识别 竞赛 - DataFountain)

由于竞赛平台协议有规定，未经许可暂不能公开数据集。

2、yolov5地址[YOLOv5](ultralytics/yolov5).

# 环境

Windows10系统

显卡：RTX 3060 12G显存

cuda:11.1

pytorch:1.8.0

其他所需项可以参考文件**requirements**内容。

# 步骤

## Step1：准备自己的数据集

新建steel_data数据集，结构如下：

    -steel_data
      -Annotations
      -images
      -ImageSets
        - Main
      -labels

 **说明：**
 
Annotations存放的是标签xml文件

images存放的是原始图片文件

ImageSets/Main存放的是train.txt, test.txt文件，使用split_train_val.py生成

labels把数据集格式转换成yolo_txt格式，即将每个xml标注提取bbox信息为txt格式，使用voc_label.py生成

通过上面两个文件转换，生成最终自己的数据集。

## Step2：配置文件

### 1、yaml文件

在data文件夹下新建一个steel.yaml文件，用来存放训练集和验证集的划分文件地址，目标类别和类别列表。

在创建文件时，nc那里要有空格，否则会报错。

### 2、anchors文件

运行clauculate_anchors.py文件，生成新的anchors文件。

此处需要修改成自己的文件夹路径。

生成的anchors.txt文件存放建议的先验框内容。

### 3、模型配置

选择一个需要的模型，这里有4个模型，分别为s, m, l, x版本，随着架构的增大，时间也会随之增大。这里选择使用的是s模型。将里面的nc数量和anchors进行修改。

钢筋识别时nc为1即可。

### 4、安装pycocotools

安装时需要有vs软件，否则会报错。

百度云链接：https://pan.baidu.com/s/1gxI7a_-68-NyU67wQMIDJw

提取码：tpbj

```python
cd cocoapi/PythonAPI  
python setup.py build_ext install
```

## Step3：模型训练

### 1、下载权重文件

到[此处](https://github.com/ultralytics/yolov5/releases/tag/v5.0)各个版本的权重文件

或者运行weights文件夹下的sh文件下载权重，把下载的权重放到weights文件夹下。

### 2、修改train文件

```
python train.py --img-size 640 --epochs 200 --data data/steel.yaml --batch-size 8 --weights weights/yolov5s.pt --cfg models/yolov5s.yaml --device 0
```

### 3、模型测试

在有标注的测试集或验证集上进行模型效果的评估，在test.py文件中指定配置文件和模型训练的结果，运行即可完成模型测试。

```
python test.py  --data data/steel.yaml --weights runs/exp10/weights/best.pt --augment
```

### 4、模型推理

对没有标注的文件进行推理，修改detect.py文件指定测试图片和测试模型的路径，即可完成模型推理。

```
python detect.py --weights runs/train/exp10/weights/best.pt --source inference/ --device 0 --save-txt --conf-thres 0.5
```

我这里还修改了detect中图形表示，将方框修改成了圆框，并将数字标注在了识别的结果上。



