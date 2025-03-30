# import pandas as pd
# import numpy as np
# import re
# import ast
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import nltk
# import warnings
# from weightboost import WeightBoost, AdaBoost, C45Tree 
# warnings.filterwarnings('ignore')
# np.random.seed(66)


# # 下载必要的NLTK资源
# try:
#     nltk.download('stopwords', quiet=True)
# except:
#     print("无法下载NLTK资源，请确保您已经安装了NLTK")

# def preprocess_reuters(data_path):
#     """
#     预处理 Reuters 数据集（多标签任务）
#     参数：
#         data_path (str): 输入的 Excel 文件路径
#     返回：
#         X_train, y_train, X_test, y_test: 预处理后的训练集和测试集
#     """
#     # 1. 读取数据
#     print("读取数据...")
#     df = pd.read_excel(data_path)
#     df['text'] = df['text'].fillna("")  # 确保文本列没有空值
#     df['categories'] = df['categories'].fillna("[]")  # 确保类别列没有空值

#     # 将字符串形式的类别解析为 Python 列表
#     print("解析类别列...")
#     df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

#     # 2. 文本预处理
#     print("进行文本清理...")
#     stop_words = set(stopwords.words("english"))  # 停用词
#     stemmer = PorterStemmer()  # 词干提取器

#     def tokenize_and_clean(text):
#         # 转小写
#         text = text.lower()
#         # 去除标点符号
#         text = re.sub(r"[^\w\s]", "", text)
#         # 分词并去除停用词，进行词干提取
#         tokens = text.split()
#         tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
#         return " ".join(tokens)

#     # 对所有文本进行清理
#     df['cleaned_text'] = df['text'].apply(tokenize_and_clean)

#     # 3. 特征选择：使用 TF-IDF 向量化
#     print("进行特征选择...")
#     vectorizer = TfidfVectorizer(max_features=2000)  # 限制最多 2000 个特征
#     X = vectorizer.fit_transform(df['cleaned_text'])  # 文档向量化
#     X = X.toarray()  # Convert to dense array for C45Tree

#     # 提取类别标签并进行多标签编码
#     print("进行多标签编码...")
#     mlb = MultiLabelBinarizer()
#     y = mlb.fit_transform(df['categories'])  # 将类别转换为多标签 one-hot 编码

#     # 将 MultiLabelBinarizer 类别保存为属性，便于后续解码
#     preprocess_reuters.mlb_classes = mlb.classes_

#     # 4. 数据集划分
#     print("划分训练集和测试集...")
#     train_indices = df[df['ids'].str.startswith('training')].index
#     test_indices = df[df['ids'].str.startswith('test')].index

#     X_train, X_test = X[train_indices], X[test_indices]
#     y_train, y_test = y[train_indices], y[test_indices]

#     print("预处理完成！")
#     return X_train, y_train, X_test, y_test, mlb.classes_

# def convert_labels_to_binary(y, positive=1, negative=-1):
#     """
#     将0/1标签转换为-1/1标签，适用于AdaBoost和WeightBoost
#     """
#     return np.where(y == 1, positive, negative)

# '''
# # C45Tree wrapper class to match sklearn interface for boosting
# class C45Wrapper:
#     def __init__(self, criterion='entropy'):
#         self.tree = C45Tree(min_samples_split=2, max_depth=None)
        
#     def fit(self, X, y):
#         self.tree.fit(X, y)
#         return self
    
#     def predict(self, X):
#         return self.tree.predict(X)
# '''

# def main():
#     # 数据路径
#     data_path = "reutersNLTK.xlsx"
    
#     # 预处理数据
#     X_train, y_train, X_test, y_test, categories = preprocess_reuters(data_path)
    
#     print(f"训练集大小: {X_train.shape}")
#     print(f"测试集大小: {X_test.shape}")
#     print(f"类别数量: {y_train.shape[1]}")
    
#     # 找出所有目标类别的索引
#     target_categories = ['trade', 'grain', 'crude', 'corn', 'ship', 'wheat', 'acq', 'interest', 'money-fx', 'earn']
#     target_indices = []
    
#     for cat in target_categories:
#         for i, category in enumerate(categories):
#             if cat.lower() == category.lower():
#                 target_indices.append(i)
#                 break
    
#     if len(target_indices) != len(target_categories):
#         print(f"警告：未找到所有目标类别。找到了 {len(target_indices)}/{len(target_categories)} 个类别")
    
#     print(f"选择的类别: {[categories[i] for i in target_indices]}")
    
#     # 准备结果表格
#     results = []
    
#     for i, cat_idx in enumerate(target_indices):
#         cat_name = categories[cat_idx]
#         print(f"\n处理类别 {cat_name} ({i+1}/{len(target_indices)})...")
        
#         # 获取当前类别的标签
#         y_train_cat = y_train[:, cat_idx]
#         y_test_cat = y_test[:, cat_idx]
        
#         # 将标签转换为 {-1, 1} 格式，适用于AdaBoost和WeightBoost
#         y_train_binary = convert_labels_to_binary(y_train_cat)
#         y_test_binary = convert_labels_to_binary(y_test_cat)
        
#         # 训练C4.5决策树（基准）
#         print("训练C4.5决策树...")
#         c45 = C45Tree(min_samples_split=2, max_depth=5)
#         c45.fit(X_train, y_train_cat)
#         y_pred_c45 = c45.predict(X_test)
#         f1_c45 = f1_score(y_test_cat, y_pred_c45)
        
#         # 训练AdaBoost
#         print("训练AdaBoost...")
#         ada = AdaBoost(n_estimators=50)
#         ada.fit(X_train, y_train_binary)
#         y_pred_ada = ada.predict(X_test)
#         # 将-1/1预测转换回0/1以计算F1分数
#         y_pred_ada_01 = np.where(y_pred_ada == 1, 1, 0)
#         f1_ada = f1_score(y_test_cat, y_pred_ada_01)
        
#         # 计算AdaBoost相对于C4.5的改进
#         ada_impro = ((f1_ada - f1_c45) / f1_c45) * 100 if f1_c45 > 0 else 0
        
#         # 训练WeightBoost
#         print("训练WeightBoost...")
#         wb = WeightBoost(
#             n_estimators=50,
#             beta=0.5
#         )
#         wb.fit(X_train, y_train_binary)
#         y_pred_wb = wb.predict(X_test)
#         # 将-1/1预测转换回0/1以计算F1分数
#         y_pred_wb_01 = np.where(y_pred_wb == 1, 1, 0)
#         f1_wb = f1_score(y_test_cat, y_pred_wb_01)
        
#         # 计算WeightBoost相对于C4.5的改进
#         wb_impro = ((f1_wb - f1_c45) / f1_c45) * 100 if f1_c45 > 0 else 0
        
#         # 添加结果
#         results.append({
#             'Category': cat_name,
#             'C4.5_F1': f1_c45,
#             'AdaBoost_F1': f1_ada,
#             'AdaBoost_Impro': f"{ada_impro:.1f}%",
#             'WeightBoost_F1': f1_wb,
#             'WeightBoost_Impro': f"{wb_impro:.1f}%"
#         })
    
#     # 创建结果DataFrame
#     results_df = pd.DataFrame(results)
    
#     # 确保类别按照指定顺序排序
#     cat_order = [c.lower() for c in target_categories]
    
#     # 创建一个映射字典来记录每个类别的位置
#     order_dict = {cat: i for i, cat in enumerate(cat_order)}
    
#     # 按照指定的顺序重新排序结果
#     results_df['order'] = results_df['Category'].apply(lambda x: order_dict.get(x.lower(), len(cat_order)))
#     results_df = results_df.sort_values('order').drop('order', axis=1)
    
#     # 格式化F1分数为4位小数
#     results_df['C4.5_F1'] = results_df['C4.5_F1'].apply(lambda x: f"{x:.4f}")
#     results_df['AdaBoost_F1'] = results_df['AdaBoost_F1'].apply(lambda x: f"{x:.4f}")
#     results_df['WeightBoost_F1'] = results_df['WeightBoost_F1'].apply(lambda x: f"{x:.4f}")
    
#     # 打印结果表格
#     print("\n表格结果:")
#     print(results_df[['Category', 'C4.5_F1', 'AdaBoost_F1', 'AdaBoost_Impro', 'WeightBoost_F1', 'WeightBoost_Impro']].to_string(index=False))
    
#     # 保存结果到CSV
#     results_df.to_csv('reuters_boosting_results.csv', index=False)
#     print("\n结果已保存到 reuters_boosting_results.csv")

# if __name__ == "__main__":
#     main()




import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import warnings
from weightboost import WeightBoost, AdaBoost, C45Tree
warnings.filterwarnings('ignore')
np.random.seed(123)


# 下载必要的NLTK资源
try:
    nltk.download('stopwords', quiet=True)
except:
    print("无法下载NLTK资源，请确保您已经安装了NLTK")

def preprocess_reuters(data_path):
    """
    预处理 Reuters 数据集（多标签任务）
    参数：
        data_path (str): 输入的 Excel 文件路径
    返回：
        X_train, y_train, X_test, y_test: 预处理后的训练集和测试集
    """
    # 1. 读取数据
    print("读取数据...")
    df = pd.read_excel(data_path)
    df['text'] = df['text'].fillna("")  # 确保文本列没有空值
    df['categories'] = df['categories'].fillna("[]")  # 确保类别列没有空值

    # 将字符串形式的类别解析为 Python 列表
    print("解析类别列...")
    df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 2. 文本预处理
    print("进行文本清理...")
    stop_words = set(stopwords.words("english"))  # 停用词
    stemmer = PorterStemmer()  # 词干提取器

    def tokenize_and_clean(text):
        # 转小写
        text = text.lower()
        # 去除标点符号
        text = re.sub(r"[^\w\s]", "", text)
        # 分词并去除停用词，进行词干提取
        tokens = text.split()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    # 对所有文本进行清理
    df['cleaned_text'] = df['text'].apply(tokenize_and_clean)

    # 3. 特征选择：使用 TF-IDF 向量化
    print("进行特征选择...")
    vectorizer = TfidfVectorizer(max_features=2000)  # 限制最多 2000 个特征
    X = vectorizer.fit_transform(df['cleaned_text'])  # 文档向量化
    X = X.toarray()  # Convert to dense array for C45Tree

    # 提取类别标签并进行多标签编码
    print("进行多标签编码...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])  # 将类别转换为多标签 one-hot 编码

    # 将 MultiLabelBinarizer 类别保存为属性，便于后续解码
    preprocess_reuters.mlb_classes = mlb.classes_

    # 4. 数据集划分
    print("划分训练集和测试集...")
    train_indices = df[df['ids'].str.startswith('training')].index
    test_indices = df[df['ids'].str.startswith('test')].index

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print("预处理完成！")
    return X_train, y_train, X_test, y_test, mlb.classes_

def convert_labels_to_binary(y, positive=1, negative=-1):
    """
    将0/1标签转换为-1/1标签，适用于AdaBoost和WeightBoost
    """
    return np.where(y == 1, positive, negative)

def main():
    # 数据路径
    data_path = "reutersNLTK.xlsx"
    
    # 预处理数据
    X_train, y_train, X_test, y_test, categories = preprocess_reuters(data_path)
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"类别数量: {y_train.shape[1]}")
    
    # 选择10个最常见的类别（保留原有代码）
    category_counts = np.sum(y_train, axis=0)
    top_10_indices = np.argsort(category_counts)[-10:]
    top_10_categories = categories[top_10_indices]
    
    print(f"选择的10个最常见类别: {top_10_categories}")
    
    # 准备结果表格
    results = []
    
    for i, cat_idx in enumerate(top_10_indices):
        cat_name = categories[cat_idx]
        print(f"\n处理类别 {cat_name} ({i+1}/10)...")
        
        # 获取当前类别的标签
        y_train_cat = y_train[:, cat_idx]
        y_test_cat = y_test[:, cat_idx]
        
        # 将标签转换为 {-1, 1} 格式，适用于AdaBoost和WeightBoost
        y_train_binary = convert_labels_to_binary(y_train_cat)
        y_test_binary = convert_labels_to_binary(y_test_cat)
        
        # 训练C4.5决策树（基准）
        print("训练C4.5决策树...")
        c45 = C45Tree(min_samples_split=2, max_depth=5)
        c45.fit(X_train, y_train_cat)
        y_pred_c45 = c45.predict(X_test)
        f1_c45 = f1_score(y_test_cat, y_pred_c45)
        
        # 训练AdaBoost
        print("训练AdaBoost...")
        ada = AdaBoost(
            n_estimators=50
        )
        ada.fit(X_train, y_train_binary)
        y_pred_ada = ada.predict(X_test)
        # 将-1/1预测转换回0/1以计算F1分数
        y_pred_ada_01 = np.where(y_pred_ada == 1, 1, 0)
        f1_ada = f1_score(y_test_cat, y_pred_ada_01)
        
        # 计算AdaBoost相对于C4.5的改进
        ada_impro = ((f1_ada - f1_c45) / f1_c45) * 100 if f1_c45 > 0 else 0
        
        # 训练WeightBoost
        print("训练WeightBoost...")
        wb = WeightBoost(
            n_estimators=50,
            beta=0.5
        )
        wb.fit(X_train, y_train_binary)
        y_pred_wb = wb.predict(X_test)
        # 将-1/1预测转换回0/1以计算F1分数
        y_pred_wb_01 = np.where(y_pred_wb == 1, 1, 0)
        f1_wb = f1_score(y_test_cat, y_pred_wb_01)
        
        # 计算WeightBoost相对于C4.5的改进
        wb_impro = ((f1_wb - f1_c45) / f1_c45) * 100 if f1_c45 > 0 else 0
        
        # 添加结果
        results.append({
            'Category': cat_name,
            'C4.5_F1': f1_c45,
            'AdaBoost_F1': f1_ada,
            'AdaBoost_Impro': f"{ada_impro:.1f}%",
            'WeightBoost_F1': f1_wb,
            'WeightBoost_Impro': f"{wb_impro:.1f}%"
        })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 指定类别顺序
    desired_order = ['trade', 'grain', 'crude', 'corn', 'ship', 'wheat', 'acq', 'interest', 'money-fx', 'earn']
    
    # 创建一个排序键，将类别名转为小写并与所需顺序比较
    results_df['sort_key'] = results_df['Category'].apply(
        lambda x: desired_order.index(x.lower()) if x.lower() in desired_order else len(desired_order)
    )
    
    # 按照指定顺序排序
    results_df = results_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # 格式化F1分数为4位小数
    results_df['C4.5_F1'] = results_df['C4.5_F1'].apply(lambda x: f"{x:.4f}")
    results_df['AdaBoost_F1'] = results_df['AdaBoost_F1'].apply(lambda x: f"{x:.4f}")
    results_df['WeightBoost_F1'] = results_df['WeightBoost_F1'].apply(lambda x: f"{x:.4f}")
    
    # 打印结果表格
    print("\n表格结果:")
    print(results_df[['Category', 'C4.5_F1', 'AdaBoost_F1', 'AdaBoost_Impro', 'WeightBoost_F1', 'WeightBoost_Impro']].to_string(index=False))
    
    # 保存结果到CSV
    results_df.to_csv('reuters_boosting_results.csv', index=False)
    print("\n结果已保存到 reuters_boosting_results.csv")

if __name__ == "__main__":
    main()