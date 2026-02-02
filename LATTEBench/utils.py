import os
import random
from openai import OpenAI
import time
import torch
import json
import pandas as pd
import numpy as np
import re
import shutil
from pathlib import Path

def remove_bold(line: str) -> str:
    # 去除 Markdown 粗體標記 **
    line = re.sub(r"\*\*", "", line)
    # 去除引号
    line = re.sub(r"\"", "", line)
    # 去除`
    line = re.sub(r"\`", "", line)
    return line.strip()

def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        return content

def read_txt_to_list(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除换行符和空白，按逗号分割，并去掉空项
            items = [item.strip() for item in line.strip().split(',') if item.strip()]
            result.extend(items)
    return result

def extract_json_content(raw_string):
    # 查找第一个 { 和最后一个 } 的位置
    start = raw_string.find('{')
    end = raw_string.rfind('}')
    
    if start == -1 or end == -1:
        raise ValueError("字符串中未找到有效的 {} 包裹的JSON内容")
    
    # 提取内容（包含{}）
    return raw_string[start:end+1]

def get_data_samples(df_sampled, target):
    data_samples = ""
    df_current = df_sampled.groupby(
        target, group_keys=False
    ).apply(lambda x: x.sample(frac=1))

    for _, icl_row in df_current.iterrows():
        answer = icl_row[target]
        icl_row = icl_row.drop(labels=target)
        data_samples += serialize(icl_row)
        data_samples += f"\nAnswer: {answer}\n"
    
    return data_samples

def serialize(row):
    target_str = f""
    for attr_idx, attr_name in enumerate(list(row.index)):
        if attr_idx < len(list(row.index)) - 1:
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += ". "
        else:
            # if len(attr_name.strip()) < 2:
            #     continue
            target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
            target_str += "."
    return target_str

def fill_template(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str

def query_llm(prompt, max_tokens=8000, temperature=1, model="deepseek-chat"):
    if model=="deepseek-chat" or model=="deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif model=="llama-3.1-8b-instruct" or model=="llama-3.1-70b-instruct":
        model = "meta-llama/" + model
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = 1
    )
    result = response.choices[0].message.content
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    return result, token_usage

def query_critic(last_user ,last_llm , prompt, max_tokens=8000, temperature=1, model="deepseek-chat"):
    if model=="deepseek-chat" or model=="deepseek-reasoner":
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    elif model=="llama-3.1-8b-instruct" or model=="llama-3.1-70b-instruct":
        model = "meta-llama/" + model
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": last_user},
            {"role": "assistant", "content": last_llm},
            {"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = 1
    )
    result = response.choices[0].message.content
    token_usage = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    return result, token_usage

def extract_function_body(code):
    lines = code.split('\n')
    start_line = None
    end_line = None
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('def feature_generation(df):'):
            start_line = i + 1
        elif start_line is not None and stripped_line.startswith('return df'):
            end_line = i
            break
    if start_line is None or end_line is None:
        return ""
    return lines[start_line:end_line]

def copy_and_rename_metadata(data_name):
    src_file = Path(f"./metadata/{data_name}.json")
    dst_dir = Path(f"./tmp/{data_name}")
    dst_file = dst_dir / "metadata.json"

    # 确保源文件存在
    if not src_file.exists():
        raise FileNotFoundError(f"源文件不存在: {src_file}")

    # 创建目标目录（如果不存在）
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 拷贝并重命名
    shutil.copy(src_file, dst_file)
    print(f"已将 {src_file} 拷贝到 {dst_file}")

def clear_files(directory):
    """
    清空指定目录及其子目录下与 filenames 中指定名称相符的文件内容。
    
    :param directory: 目标文件夹路径
    :param filenames: 要清空的文件名列表（精确匹配）
    """
    if not os.path.isdir(directory):
        print(f"目录不存在: {directory}")
        return
    filenames = ["best_train.csv","best_test.csv","feature_generation.py","full_code.py","metadata.json","train.csv","val.csv","test.csv"]
    cleared_files = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file in filenames:
                file_path = os.path.join(root, file)
                try:
                    open(file_path, 'w').close()
                    print(f"已清空: {file_path}")
                    cleared_files += 1
                except Exception as e:
                    print(f"无法清空 {file_path}: {e}")

    print(f"共清空 {cleared_files} 个文件。")

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并删除DataFrame中重复的列名，只保留第一次出现的列。
    
    参数:
        df (pd.DataFrame): 输入的DataFrame
    
    返回:
        pd.DataFrame: 删除重复列后的DataFrame
    """
    # 利用pandas的Index.duplicated方法检测重复列名
    return df.loc[:, ~df.columns.duplicated()]
