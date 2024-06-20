import pandas as pd
import numpy as np

# 读取Excel文件中的所有sheet的数据和sheetname
file_path = r"**.xlsx"
xl = pd.ExcelFile(file_path)

# 创建一个空的列表用于存储处理后的数据
data = []

# 遍历所有sheet
for sheet_name in xl.sheet_names:
    # 读取每个sheet的数据
    sheet_data = xl.parse(sheet_name)
    
    # 获取第三列和第四列的数据
    col_3 = sheet_data.iloc[:, 2]  # 第三列的数据
    col_4 = sheet_data.iloc[:, 3]  # 第四列的数据
    
    # 找出第四列大于第四列最大值的0.1的部分对应的第三列数据
    filtered_col_3 = col_3[col_4 > col_4.max() * 0.1]

    # 计算曲线与 x 轴围成的面积（数值积分）
    area_under_curve = np.trapz(filtered_col_3, col_4[filtered_col_3.index])
    col_4_max_value = sheet_data.iloc[:, 3].max()  # 第四列的最大值
    
    # 创建包含提取的值和积分结果的列表，以准备转换为DataFrame的一行
    row_data = [sheet_name, col_4_max_value, area_under_curve]
    # 将数据添加到列表中
    data.append(row_data)
    
# 创建DataFrame
columns = ['SheetName', 'Tensile_Strength', 'Area_Under_Curve']
result_df = pd.DataFrame(data, columns=columns)

# 将DataFrame保存到data.xlsx文件中
output_file_path =r"**.xlsx"
result_df.to_excel(output_file_path, index=False)
