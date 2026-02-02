import pandas as pd
import os
import json
import utils
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from Evaluator import safe_scale_large_values

def calculate_rf_importance(train_data, test_data, model, target, task_type):
    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    # 预处理
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            # 填充缺失值
            X_train[col] = X_train[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")
            # Label Encoding（训练和测试统一编码）
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
        
        elif X_train[col].dtype == "category":
            # 填充缺失值
            X_train[col] = X_train[col].cat.add_categories(["missing"]).fillna("missing")
            X_test[col] = X_test[col].cat.add_categories(["missing"]).fillna("missing")
            # Label Encoding（训练和测试统一编码）
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        
        else:
            # 数值列填充缺失值（用中位数更稳健）
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)

    X_train, X_test = safe_scale_large_values(X_train, X_test)

    if task_type == 1:
        r = permutation_importance(model, X_test, y_test,
                            n_repeats=5,
                            random_state=0)
    elif task_type == 0:
        r = permutation_importance(model, X_test, y_test,
                            n_repeats=5, scoring='neg_root_mean_squared_error',
                            random_state=0)
    
    return r.importances_mean

def select_top_features(df, importance, k: int):
    # 构建特征重要性 DataFrame
    feature_importance = pd.DataFrame({
        "feature": df.columns[:-1],   # 假设最后一列是目标列
        "importance": importance
    })
    # 去掉 importance < 0 的特征
    feature_importance = feature_importance[feature_importance["importance"] >= 0]
    # 按重要性排序
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    # 选出前k个特征
    top_features = feature_importance.head(k)["feature"].tolist()
    # 确保目标列保留
    selected_columns = top_features + ['target']
    # 被删除的列（除了目标列）
    dropped_features = [col for col in df.columns if col not in selected_columns]

    return df[selected_columns].copy(), dropped_features

def keep_most_important_features(
    df: pd.DataFrame,
    test_importance: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    # 过滤掉 importance 低于阈值的特征
    filtered_importance = test_importance[test_importance['importance'] >= 0.0]

    # 按照 importance 降序排序，并选出最多 k 个特征名
    most_important_features = (
        filtered_importance.sort_values(by='importance', ascending=False)
        .head(k).index.tolist()
    )

    # 从原始 df 中选择这些列（如果存在）
    df_filtered = df[[col for col in most_important_features if col in df.columns]].copy()

    # 被删除的columns列表
    dropped_columns = [col for col in df.columns if col not in df_filtered.columns]

    return df_filtered, dropped_columns
    

def update_metadata(metadata, dropped_columns):
    # 删除指定键
    for col in dropped_columns:
        metadata.pop(col, None)  # 用 pop 避免不存在时报错

    return metadata

def update_code(dropped_columns, data_name):
    code_path = os.path.join('tmp', data_name, 'feature_generation.py')
    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()

    old_line = utils.extract_function_body(code)
    
    code_path = os.path.join('tmp', data_name, 'full_code.py')

    # 读取代码
    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()

    lines = code.split('\n')
    
    # 计算缩进（保持一致风格）
    indent = ' ' * 4
    
    # 插入 drop 行
    drop_line = [(
        indent + f"df.drop(columns={dropped_columns}, inplace=True)"
    )]

    if dropped_columns == []:
        lines = old_line + lines
    else:
        lines = old_line + lines + drop_line

    # 重新拼接代码
    updated_code = '\n'.join(lines)

    # 保存回原文件
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(updated_code)

    return updated_code