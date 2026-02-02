import zss  # pip install zss

# ========= 树节点 =========
class Node(zss.Node):
    def __init__(self, label, children=None):
        super().__init__(label)
        if children:
            for c in children:
                self.addkid(c)

# ========= 后缀表达式 → 树 =========
def postfix_to_tree(expr):
    tokens = expr.split()
    if len(tokens) == 1 and tokens[0].startswith("f"):  # 原子特征
        return Node(tokens[0])

    stack = []
    for t in tokens:
        if t.startswith("f"):
            stack.append(Node(t))
        else:  # 二元或一元运算符
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                stack.append(Node(t, [left, right]))
            elif len(stack) == 1:  # 单目运算
                child = stack.pop()
                stack.append(Node(t, [child]))
            else:
                raise ValueError(f"Invalid postfix expression: {expr}")
    if len(stack) != 1:
        raise ValueError(f"Invalid postfix expression: {expr}")
    return stack[0]

# ========= 树编辑距离 =========
def tree_edit_distance(t1, t2):
    return zss.simple_distance(
        t1, t2,
        get_children=lambda n: n.children,
        get_label=lambda n: n.label,
        insert_cost=lambda n: 1,
        remove_cost=lambda n: 1,
        update_cost=lambda a, b: 0 if a == b else 1
    )

def tree_size(t):
    return 1 + sum(tree_size(c) for c in t.children)

# ========= 两个表达式列表的相似度 =========
def expr_list_similarity(exprs1, exprs2):
    sims = []
    for expr1 in exprs1:
        t1 = postfix_to_tree(expr1)
        best_sim = 0
        for expr2 in exprs2:
            t2 = postfix_to_tree(expr2)
            dist = tree_edit_distance(t1, t2)
            max_size = max(tree_size(t1), tree_size(t2))
            sim = 1 - dist / max_size
            best_sim = max(best_sim, sim)
        sims.append(best_sim)
    return sum(sims) / len(sims) if sims else 1.0

# # ========= 示例 =========
# if __name__ == "__main__":
#     exprs1 = ["f1", "f2", "f3", "f4", "f3 square"]
#     exprs2 = ["f1", "f2", "f3", "f4", "f3 square", "f3 reciprocal"]

#     sim = expr_list_similarity(exprs1, exprs2)
#     print("Similarity:", sim)



def init_feature_map(metadata):
    """
    根据初始metadata生成特征映射
    metadata: dict, e.g. {"Airline":"Airline","Flight":"Flight","Time":"Time","Length":"Length"}
    return: dict {特征名: 后缀表达式}
    """
    feature_map = {}
    for idx, col in enumerate(metadata.keys(), start=1):
        feature_map[col] = f"f{idx}"  # 初始特征直接映射为 f1,f2,...
    return feature_map


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def update_feature_map(feature_map, operations):
    """
    根据操作列表更新特征映射（支持链式引用，多轮更新）
    feature_map: dict, 当前特征映射，{特征名: 后缀表达式}
    operations: list[dict], 每个新特征定义
    return: dict, 更新后的特征映射
    """
    for op in operations:

        new_name = op["new_feature_name"]
        operator = op["operator"]
        f1 = op["feature1"]
        f2 = op.get("feature2")

        # 确保输入特征存在
        if f1 not in feature_map:
            raise ValueError(f"Feature {f1} not found")
        if f2 and f2 not in feature_map and is_number(f2)==False:
            raise ValueError(f"Feature {f2} not found")

        # 完全展开
        expr1 = feature_map[f1]
        if f2:
            if is_number(f2)==False:
                expr2 = feature_map[f2]
            else:
                expr2 = op.get("feature2")
        else:
            expr2 = None

        if expr2:
            postfix = f"{expr1} {expr2} {operator}"
        else:
            postfix = f"{expr1} {operator}"

        feature_map[new_name] = postfix

    return feature_map

def metadata_to_expr_list(metadata, feature_map):
    """
    根据metadata和特征映射生成后缀表达式列表。

    Parameters
    ----------
    metadata : dict
        其键为列/特征名（顺序即输出顺序），值内容无关。
    feature_map : dict[str, str]
        {特征名: 后缀表达式或 'fN'}，要求包含 metadata 中的所有键。

    Returns
    -------
    list[str]
        与 metadata 键一一对应的后缀表达式（或单个 fN）。

    Raises
    ------
    KeyError
        当 metadata 的某个键不在 feature_map 中时抛出。
    """
    expr_list = []
    for key in metadata.keys():  # 依 metadata 键的插入顺序
        try:
            expr = feature_map[key]
        except KeyError:
            raise KeyError(f"metadata key '{key}' not found in feature_map")
        expr_list.append(expr)
    return expr_list


def expr_list_to_feature_names(expr_list, metadata):
    """
    将后缀表达式列表中的f_n翻译回初始特征名称。

    Parameters
    ----------
    expr_list : list[str]
        后缀表达式列表，包含f1, f2等特征引用。
    metadata : dict
        初始的metadata字典，其键为特征名，顺序与f1, f2, f3对应。

    Returns
    -------
    list[str]
        将f_n替换为对应特征名称后的表达式列表。

    Raises
    ------
    ValueError
        当后缀表达式中包含超出metadata范围的f_n时抛出。
    """
    # 创建从f_n到特征名称的映射
    f_to_feature = {}
    for idx, feature_name in enumerate(metadata.keys(), start=1):
        f_to_feature[f"f{idx}"] = feature_name
    
    result = []
    for expr in expr_list:
        # 将表达式按空格分割
        tokens = expr.split()
        translated_tokens = []
        
        for token in tokens:
            if token.startswith("f") and token[1:].isdigit():
                # 这是一个f_n格式的特征引用
                if token in f_to_feature:
                    translated_tokens.append(f_to_feature[token])
                else:
                    raise ValueError(f"Feature reference '{token}' not found in metadata")
            else:
                # 这是运算符或其他内容，保持不变
                translated_tokens.append(token)
        
        # 重新组合为表达式
        translated_expr = " ".join(translated_tokens)
        result.append(translated_expr)
    
    return result


## ========= 示例 =========
# if __name__ == "__main__":
#     # 示例metadata
#     metadata = {"Airline":"Airline","Flight":"Flight","Time":"Time","Length":"Length"}
    
#     # 初始化特征映射
#     feature_map = init_feature_map(metadata)
#     print("初始特征映射:", feature_map)
    
#     # 将metadata转换为后缀表达式列表
#     expr_list = metadata_to_expr_list(metadata, feature_map)
#     print("后缀表达式列表:", expr_list)
    
#     # 将后缀表达式翻译回特征名称
#     feature_names = expr_list_to_feature_names(expr_list, metadata)
#     print("翻译回特征名称:", feature_names)
    
#     # 测试复杂表达式
#     complex_exprs = ["f1 f2 add", "f3 square", "f1 f2 multiply f3 divide"]
#     translated_complex = expr_list_to_feature_names(complex_exprs, metadata)
#     print("复杂表达式翻译:", translated_complex)
