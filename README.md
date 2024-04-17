import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
# 导入pycaret库
import pycaret

# 打印pycaret库的版本号
pycaret.__version__


data = pd.read_csv('./analysisData.csv')

# 导入pycaret.regression模块中的所有函数
from pycaret.regression import *

# 初始化设置
# data为数据集，target为目标变量
# s为初始化后的设置，包含了数据预处理、特征工程、模型选择等步骤
s = setup(data, target='price', categorical_features = [
    'make_name'
    # , 'model_name', 'trim_name', 'body_type', 'fuel_type'
])

# 导入必要的库
import torch
from transformers import AutoModel, AutoTokenizer

best = compare_models()  # 调用compare_models()函数，并将返回的最佳模型赋值给best_model变量。
print(best)

# 预测保留集上的结果
pred_holdout = predict_model(best)


data2 = pd.read_csv('./scoringData.csv')
# 生成预测
predictions = predict_model(best, data=data2)  # 使用best_model对data2进行预测，将结果保存在predictions中
print(predictions)

predictions.to_csv('./predictions_filtered.csv', index=False)
# # 选择'id'和'price'列
# predictions_filtered = predictions[['id', 'price']]

# # 将 DataFrame 保存为 CSV 文件
# predictions_filtered.to_csv('./predictions_filtered.csv', index=False)
