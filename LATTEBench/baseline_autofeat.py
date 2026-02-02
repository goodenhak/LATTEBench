import os
import sys
import warnings
import Evaluator
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from autofeat import AutoFeatClassifier

import os
import logging
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='AutoFeat Feature Engineering Pipeline')
    
    # 日志相关参数
    parser.add_argument('--log_path', type=str, default='./log', 
                       help='Path to log directory')
    parser.add_argument('--log_filename', type=str, default=None,
                       help='Log filename (default: {data_name}_AutoFeat.log)')
    
    # 主要参数
    parser.add_argument('--data_name', type=str, default='credit-g',
                       help='Dataset name')
    parser.add_argument('--seed', type=int, default=2,
                       help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')
    
    # AutoFeat参数
    parser.add_argument('--feateng_steps', type=int, default=2,
                       help='Feature engineering steps for AutoFeat')
    parser.add_argument('--max_gb', type=int, default=1,
                       help='Maximum memory usage in GB for AutoFeat')
    parser.add_argument('--n_jobs', type=int, default=10,
                       help='Number of parallel jobs for AutoFeat')
    
    return parser.parse_args()

def setup_logging(log_path, log_filename):
    """设置日志配置"""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 确保日志目录存在
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, log_filename)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    return logger

def main():
    # 解析命令行参数
    args = parse_arguments()
    task_type = 1

    # 设置日志文件名（如果没有指定则使用数据集名称）
    if args.log_filename is None:
        args.log_filename = f"{args.data_name}_AutoFeat_{args.seed}.log"
    
    # 设置日志
    logger = setup_logging(args.log_path, args.log_filename)
    logger.info("========== START ==========")
    logger.info(f"Arguments: {vars(args)}")
    
    # 开始计时
    total_start_time = time.time()
    
    # 读取并预处理数据
    csv_file_path = f'./tabular_data/{args.data_name}.csv'
    logger.info(f"Reading CSV file: '{csv_file_path}'...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, encoding='utf-8', low_memory=False)
    logger.info(f"Original data shape: {df.shape}")

    # 分离特征和目标变量
    target_col = 'class'
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(target_col, axis=1)
    else:
        # 如果没有'class'列，假设最后一列是目标变量
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        target_col = df.columns[-1]

    logger.info(f"Target column: '{target_col}'")
    logger.info(f"Features shape: {X.shape}")

    # 识别数值列和分类列
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Numerical columns: {len(numerical_cols)}")
    logger.info(f"Categorical columns: {len(categorical_cols)}: {categorical_cols}")

    # 处理缺失值
    if X.isnull().sum().sum() > 0:
        logger.info("Handling missing values...")
        # 数值列用中位数填充
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 分类列用众数填充
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

    # 处理分类变量 - 标签编码
    if categorical_cols:
        logger.info("Encoding categorical columns...")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # 处理目标变量编码（如果是字符串类型）
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        logger.info("Encoding target variable...")
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))

    logger.info(f"Final preprocessed data shape: {X.shape}")
    logger.info(f"Missing values: {X.isnull().sum().sum()}")

    # 按4:1划分训练+验证集和测试集
    logger.info(f"Splitting data into train+val ({1-args.test_size}) and test ({args.test_size}) sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y  # 分层抽样，保持类别比例
    )

    # 将训练+验证集进一步划分为训练集和验证集
    val_ratio = args.val_size / (1 - args.test_size)  # 计算验证集占训练+验证集的比例
    logger.info(f"Splitting train+val set into train ({1-val_ratio}) and val ({val_ratio}) sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=y_train_val  # 分层抽样，保持类别比例
    )

    logger.info("数据预处理和划分完成！")

    # 合并特征和标签为完整的数据集
    train_val_data = X_train_val.copy()
    train_val_data[target_col] = y_train_val

    # 创建训练集DataFrame
    train_data = X_train.copy()
    train_data[target_col] = y_train

    # 创建验证集DataFrame
    val_data = X_val.copy()
    val_data[target_col] = y_val

    # 创建测试集DataFrame
    test_data = X_test.copy()
    test_data[target_col] = y_test

    logger.info("数据集合并完成！")

    # 使用AutoFeat进行特征工程
    logger.info("Starting AutoFeat feature engineering...")

    # 分离训练集的特征和标签用于AutoFeat
    X_train_autofeat = train_val_data.drop(target_col, axis=1)
    y_train_autofeat = train_val_data[target_col]

    # 初始化AutoFeatClassifier
    afc = AutoFeatClassifier(
        verbose=1,
        feateng_steps=args.feateng_steps,  # 特征工程步数
        max_gb=args.max_gb,  # 最大内存使用量(GB)
        n_jobs=args.n_jobs,  # 并行作业数
    )

    # 在训练集上拟合AutoFeat并生成新特征
    logger.info("Fitting AutoFeat on training data...")
    X_train_new = afc.fit_transform(X_train_autofeat, y_train_autofeat)

    logger.info("AutoFeat feature engineering completed.")
    total_end_time = time.time()
    # 将新特征应用到训练集
    logger.info("Transforming training data...")
    X_train_original = train_data.drop(target_col, axis=1)
    X_train_new = afc.transform(X_train_original)

    # 将新特征应用到验证集
    logger.info("Transforming validation data...")
    X_val_original = val_data.drop(target_col, axis=1)
    X_val_new = afc.transform(X_val_original)

    # 将新特征应用到测试集
    logger.info("Transforming test data...")
    X_test_original = test_data.drop(target_col, axis=1)
    X_test_new = afc.transform(X_test_original)

    # 创建包含新特征的完整数据集
    logger.info("Creating new datasets with engineered features...")

    # 训练集
    train_data_new = pd.DataFrame(X_train_new)
    train_data_new[target_col] = train_data[target_col].values

    # 验证集
    val_data_new = pd.DataFrame(X_val_new)
    val_data_new[target_col] = val_data[target_col].values

    # 测试集
    test_data_new = pd.DataFrame(X_test_new)
    test_data_new[target_col] = test_data[target_col].values

    logger.info(f"Original feature count: {X_train.shape[1]}")
    logger.info(f"New feature count: {X_train_new.shape[1]}")
    
    # 评估模型性能
    logger.info("Evaluating with AutoGluon...")
    predictor, val_acc = Evaluator.train_and_evaluate(train_data_new, val_data_new, target_col, task_type)
    test_acc = predictor.evaluate(test_data_new)['accuracy']
    logger.info(f"ag_acc = {test_acc}")

    logger.info("Evaluating with RandomForest...")
    predictor, val_acc = Evaluator.train_and_evaluate_rf(train_data_new, val_data_new, target_col, task_type)
    predictor, test_acc = Evaluator.train_and_evaluate_rf(train_data_new, test_data_new, target_col, task_type)
    
    logger.info(f"val_acc = {val_acc}")
    logger.info(f"test_acc = {test_acc}")
    
    # 保存结果
    output_dir = f"./tmp/{args.data_name}/"
    os.makedirs(output_dir, exist_ok=True)
    
    train_data_new.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_data_new.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_data_new.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # 记录时间
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")

if __name__ == "__main__":
    main()