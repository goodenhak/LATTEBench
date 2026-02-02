# preprocess datasets (fill missing values, and split into train, validation, and test sets)
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_and_prepare_data(data_name, task_type, seed, test_size=0.2, val_size=None):
    # 读取数据
    df = pd.read_csv(os.path.join('tabular_data', data_name + '.csv'))

    # 获取目标列名并重命名为 "target"
    original_target = df.columns[-1]
    df = df.rename(columns={original_target: "target"})

    if task_type == 1:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["target"])

    os.makedirs(os.path.join('tmp', data_name), exist_ok=True)
    df.to_csv(os.path.join('tmp', data_name, data_name + '.csv'), index=False)
    target = df.columns[-1]

    # 分离特征和目标变量
    X = df.convert_dtypes()
    y = df[target].to_numpy()
    X = X.drop(target, axis=1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if task_type == 1 else None
    )

    # 如果需要划分验证集
    if val_size is not None:
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_train if task_type == 1 else None
        )

    # 获取标签列表（仅分类任务）
    label_list = np.unique(y).tolist() if task_type == 1 else None

    # save training set and test set
    X_train[target] = y_train
    X_train.to_csv(os.path.join('tmp', data_name, 'train.csv'), index=False)
    X_test[target] = y_test
    X_test.to_csv(os.path.join('tmp', data_name, 'test.csv'), index=False)
    if val_size is not None:
        X_val[target] = y_val
        X_val.to_csv(os.path.join('tmp', data_name, 'val.csv'), index=False)
        return df, X_train, X_test, X_val, target, label_list
    return df, X_train, X_test, target, label_list