import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv('./Data/Five_year_2019_01_01-17_h_data.csv')

# 获取第二列和第三列的数据
column2 = data.iloc[:, 1]
column3 = data.iloc[:, 2]

# 绘制图形
plt.plot(column2, label='Column 2', color='blue')
plt.plot(column3, label='Column 3', color='red')

# 添加标签和标题
plt.xlabel('Row Index')
plt.ylabel('Value')
plt.title('Plot of Column 2 and Column 3')
plt.legend()

# 显示图形
plt.show()
