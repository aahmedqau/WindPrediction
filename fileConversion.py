#  文件转换，提取每天的一个小时成为新的文件

import pandas as pd

# 读取CSV文件
data = pd.read_csv('./Data/Five_year_h_data.csv')

# 定义日期解析函数
def parse_date(date_str):
    return pd.to_datetime(date_str, format='%d/%m/%Y %H').strftime('%Y-%m-%d')

# 修改日期格式
data['date'] = data['date'].apply(parse_date)

# 每隔24行取一行
filtered_data = data.iloc[12::24, :]

# 将结果存入新的CSV文件
filtered_data.to_csv('./Data/Five_year_d_12_data.csv', index=False)
