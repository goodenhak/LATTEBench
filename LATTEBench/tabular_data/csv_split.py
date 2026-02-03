import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def safe_scale_large_values(train_data: pd.DataFrame, test_data: pd.DataFrame, threshold=1e30):
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()

    train_scaled = train_scaled.replace([np.inf, -np.inf], threshold)
    test_scaled = test_scaled.replace([np.inf, -np.inf], threshold)

    max_abs = pd.concat(
        [train_scaled.abs().max(), test_scaled.abs().max()],
        axis=1
    ).max(axis=1)

    bad_cols = max_abs[max_abs > threshold].index.tolist()

    if bad_cols:
        print(f"⚠️ 检测到 {len(bad_cols)} 列值过大: {bad_cols}")
        for col in bad_cols:
            max_val = max(
                train_scaled[col].abs().max(),
                test_scaled[col].abs().max()
            )
            if max_val == 0:
                continue
            scale_factor = max_val / threshold
            train_scaled[col] /= scale_factor
            test_scaled[col] /= scale_factor
            print(f"✅ 已缩放列 '{col}'，缩放因子 = {scale_factor:.3e}")
    else:
        print("✅ 未发现超大数值或 inf，无需缩放。")

    return train_scaled, test_scaled


def split_csv(data_name, task_type, seed, test_size=0.2):
    """
    CSV -> split -> preprocess -> CSV
    """
    # ========= 读取原始 CSV =========
    df = pd.read_csv(data_name + ".csv")

    # 目标列统一为 target
    original_target = df.columns[-1]
    df = df.rename(columns={original_target: "target"})

    if task_type == 1:
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["target"])

    os.makedirs(os.path.join("tmp", data_name), exist_ok=True)
    df.to_csv(os.path.join("tmp", data_name, f"{data_name}_raw.csv"), index=False)

    target = "target"

    # ========= split =========
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if task_type == 1 else None
    )

    # 验证集
    val_ratio = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train if task_type == 1 else None
    )


    # 先保存一次（dtype 清洗）
    X_train[target] = y_train
    X_train.to_csv(os.path.join('tmp', data_name, 'train.csv'), index=False)
    X_test[target] = y_test
    X_test.to_csv(os.path.join('tmp', data_name, 'test.csv'), index=False)
    X_val[target] = y_val
    X_val.to_csv(os.path.join('tmp', data_name, 'val.csv'), index=False)

    # ========= 重新读取 =========
    train_data = TabularDataset(os.path.join("tmp", data_name, "train.csv"))
    test_data = TabularDataset(os.path.join("tmp", data_name, "test.csv"))
    val_data = TabularDataset(os.path.join("tmp", data_name, "val.csv"))

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]
    X_val = val_data.drop(columns=[target])
    y_val = val_data[target]

    # ========= 预处理 =========
    # 预处理
    for col in X_train.columns:
        if X_train[col].dtype == bool:
            X_train[col] = X_train[col].astype(np.int8)
            X_test[col]  = X_test[col].astype(np.int8)
            X_val[col]   = X_val[col].astype(np.int8)
            continue
        if X_train[col].dtype == "object":
            # 填充缺失值
            X_train[col] = X_train[col].fillna("missing")
            X_test[col] = X_test[col].fillna("missing")
            X_val[col] = X_val[col].fillna("missing")
            # Label Encoding
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            X_val[col] = le.transform(X_val[col])
        
        elif X_train[col].dtype == "category":
            # 填充缺失值
            X_train[col] = X_train[col].cat.add_categories(["missing"]).fillna("missing")
            X_test[col] = X_test[col].cat.add_categories(["missing"]).fillna("missing")
            X_val[col] = X_val[col].cat.add_categories(["missing"]).fillna("missing")
            # Label Encoding（训练和测试统一编码）
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0))
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
        
        else:
            # 数值列填充缺失值（用中位数更稳健）
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)

    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)
    # # 预测
    # y_pred = model.predict(X_val)
    # accuracy = accuracy_score(y_val, y_pred)
    # print(f"val accuracy = {accuracy}")

    # ========= 拼回 target =========
    X_train[target] = y_train
    X_test[target] = y_test
    X_val[target] = y_val

    # ========= 最终 CSV =========
    train_out = os.path.join(f"{data_name}_train.csv")
    test_out = os.path.join(f"{data_name}_test.csv")
    val_out = os.path.join(f"{data_name}_val.csv")

    X_train.to_csv(train_out, index=False)
    X_test.to_csv(test_out, index=False)
    X_val.to_csv(val_out, index=False)

    print(f"✅ 训练集已保存: {train_out}")
    print(f"✅ 测试集已保存: {test_out}")
    print(f"✅ 验证集已保存: {val_out}")


if __name__ == "__main__":
    csv_file = "wine_quality"
    task_type = 0
    seed = 2
    test_size = 0.2

    split_csv(
        data_name=csv_file,
        task_type=task_type,
        seed=seed,
        test_size=test_size
    )
