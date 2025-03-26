# How to use

## 1. Ensure that the Python environment is normal

The Python version for this project is 3.10.11

Packages matplotlib, meteostat, pandas, sklearn, numpy, torch

> ❗Please do not use numpy version 2.x or above. The version for this project is numpy1.26.1.
>
> To install sklearn, you need to install scikit-learn instead of sklearn itself

## 2. Directory introduction

main.py - main program
Data - data folder
huiTu.py - drawing program, draw data line chart without calling the main program for easy observation
fileConversion.py - file conversion program, extract some data from the original data and save it to a new file
yuan.py - project source code, easy to check

> You only need main.py and Data to start running, and there is no mutual call between the various python files

## 3. Modify parameters

#### 3.1 Modify line 35

```python
# How we read data
data = pd.read_csv('./Data/Five_year_d_12_data.csv')
```

Replace `./Data/Five_year_d_12_data.csv` with your data file

#### 3.2 Modify line 39

```python
df['tavg'] = data['A']
```

Replace `data['A']` A with the name of the data column you need, which is usually the first line in the data file

> This program can only select one data column for training at a time

#### 3.3 Modify line 366

```python
# In the following code, please note that the value of all_data should be changed to the total amount of data
all_data = 1826
```

Change the value of `all_data` to the number of data rows in the data file (if there is a table header, it should be -1, and only the data rows are counted)

#### 3.4 Modify line 344

```python
# The values ​​of df_max and df_min are abnormal and cannot be obtained. Manual rewrite
df_max = 2.21
df_min = 1.06
```

For different data, this value is usually different. You can change it as follows or reset it yourself

```python
# For this data, use the lower value when predicting annual data
df_max = 2.8
df_min = 1.5
# Use the lower value when predicting daily data A (probably this)
df_max = 2.2
df_min = 1.0
# Use the lower value when predicting daily data B
df_max = 2.81
df_min = 1.66
```

#### 3.5 Other changes

> No change is required

157 lines can customize the batch size

```python
batch_size = 64#Batch size, the stronger the computing power, the larger it can be set, customizable , the common batch size is usually between 32 and 256
```

Line 271 can change the number of training rounds

```python
num_epochs = 150
```

.....

## 4. Run

Run the main.py file, and the normal operation will output three pictures (close one and the next one will appear)

![image-20240725141038513](Readme/1.png)

![image-20240725141038513](Readme/2.png)

![image-20240725141038513](Readme/3.png)
