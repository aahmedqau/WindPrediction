# 如何使用

## 1.确保python环境正常

本项目运行时python版本为3.10.11

软件包matplotlib、meteostat、pandas、sklearn、numpy、torch

>  ❗numpy版本请勿使用2.x及以上版本、本项目运行时为numpy1.26.1。
>
> sklearn的安装需要安装scikit-learn而不是sklearn本身

## 2.目录介绍

main.py - 主程序
Data       - 数据文件夹
huiTu.py - 绘图程序，在不调用主程序的情况下绘制数据折线图方便观察
fileConversion.py - 文件转换程序，从原始数据中提取一些数据保存至新文件
yuan.py  - 项目源码，方便查验

> 你只需要有main.py 和Data就能开始运行，各个python文件间没有相互调用

## 3.修改参数

#### 3.1修改35行

```python
# 我们的数据读取方式
data = pd.read_csv('./Data/Five_year_d_12_data.csv')
```

将`./Data/Five_year_d_12_data.csv`替换你的数据文件

#### 3.2修改39行

```python
df['tavg'] = data['A']
```

将 `data['A']` A更换为你需要的数据列名,它通常是数据文件中的第一行

> 本程序一次只能选择一个数据列进行训练

#### 3.3修改366行

```python
# 下行代码中注意修改all_data值为数据量总数
all_data = 1826
```

将`all_data`值改为数据文件的数据行数(如果有表头要-1,只计算数据行)

#### 3.4修改344行

```python
# df_max与df_min值异常，无法获取。手动重写
df_max = 2.21
df_min = 1.06
```

对于不同的数据,该值通常不同,按如下更改或自行重设均可

```python
# 对于本数据预测年数据时使用下值
df_max = 2.8
df_min = 1.5
# 预测日数据A时使用下值(大概是这个)
df_max = 2.2
df_min = 1.0
# 预测日数据B时使用下值
df_max = 2.81
df_min = 1.66
```

#### 3.5其他修改

> 不改也行

157行可以自定义批量大小

```python
batch_size = 64#批量大小，算力越强，可以设置越大，可自定义 ，常见的批量大小通常在32到256之间
```

271行可以更改训练轮数

```python
num_epochs = 150
```

.....

## 4.运行

运行main.py文件,正常运行会输出三幅图(关闭一个出来下一个)

![image-20240725141038513](Readme/1.png)

![image-20240725141038513](Readme/2.png)

![image-20240725141038513](Readme/3.png)