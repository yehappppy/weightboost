# Reuters-21578 数据集

## **简介**
Reuters-21578 是一个经典的多标签文本分类数据集，广泛用于自然语言处理（NLP）任务。该数据集包含来自 Reuters 新闻社的新闻文章，每篇文章可能属于一个或多个类别。它是研究多标签分类的标准数据集之一。

### **数据集特点**
- **来源**: NLTK 的 Reuters Corpus。
- **样本数量**: 10,788 篇新闻文章。
- **类别数量**: 每篇文章可能属于 1 或多个类别，总类别数为90。
- **划分**: 数据集分为训练集和测试集（通过文件 ID 标识）。
- **形式**: 每篇新闻包含以下内容：
  - 文件 ID（`ids` 字段）
  - 文章的类别标签（`categories` 字段）
  - 文章的原始文本内容（`text` 字段）

---

## **上下文**
- **目标任务**: 数据集主要用于文本分类、多标签分类和信息检索任务。
- **结构化数据**: 每条记录代表一篇文章，包含文件 ID、类别标签和原始文本。
- **数据分布**: 数据集不均衡，少数类别（如 `earn`、`acq`）占大多数。

---
# **数据集预处理代码**

```
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ast

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
    return X_train, y_train, X_test, y_test

# Example Usage
# 输入文件路径
data_path = "./reutersNLTK.xlsx" # 替换为你的路径

# 调用预处理函数
X_train, y_train, X_test, y_test = preprocess_reuters(data_path)

# 检查预处理结果
print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])
print("特征数:", X_train.shape[1])
print("类别总数:", len(preprocess_reuters.mlb_classes))
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
print("类别列表的前十个:", preprocess_reuters.mlb_classes[:10])  # 打印前 10 个类别
```

### Expected Example Output:
```
读取数据...
解析类别列...
进行文本清理...
进行特征选择...
进行多标签编码...
划分训练集和测试集...
预处理完成！
训练集样本数: 7769
测试集样本数: 3019
特征数: 2000
类别总数: 90
X_train.shape: (7769, 2000)
X_test.shape: (3019, 2000)
y_train.shape: (7769, 90)
y_test.shape: (3019, 90)
类别列表的前十个: ['acq' 'alum' 'barley' 'bop' 'carcass' 'castor-oil' 'cocoa' 'coconut'
 'coconut-oil' 'coffee']
```


## 补充:**数据集生成代码**
以下代码用于从 NLTK 的 Reuters Corpus 中提取数据并生成一个 Pandas DataFrame：
(已执行完毕并生成`reutersNLTK.xlsx`文件，不需要再执行)

```python
import nltk
import pandas as pd
from nltk.corpus import reuters

# 下载所需的 NLTK 数据集
nltk.download('reuters')
nltk.download('punkt')

# 提取 Reuters 数据集的文件 ID
fileids = reuters.fileids()

# 初始化空列表存储类别和原始文本
categories = []
text = []

# 遍历每个文件，收集类别和原始文本
for file in fileids:
    categories.append(reuters.categories(file))  # 获取类别
    text.append(reuters.raw(file))              # 获取原始文本

# 将数据存储到 Pandas DataFrame 中
reutersDf = pd.DataFrame({'ids': fileids, 'categories': categories, 'text': text})

# 查看生成的 DataFrame
print(reutersDf.head())
```

---

## **生成的 DataFrame 示例**

以下是生成的 `reutersDf` DataFrame 的结构示例：

| **ids**         | **categories**       | **text**                                                      |
|------------------|----------------------|---------------------------------------------------------------|
| test/14826       | ['acq']             | "French electronics group Thomson-CSF..."                    |
| training/11208   | ['earn']            | "The company reported a net income of..."                    |
| training/11313   | ['acq', 'earn']     | "General Motors announced a merger with..."                  |
| test/14828       | ['crude']           | "Crude oil prices rose sharply in trading..."                |
| training/11233   | ['grain', 'corn']   | "The USDA reported higher corn yields than expected..."      |

---

## **数据集内容说明**

### **1. 字段描述**
- **`ids`**: 每篇文章的文件 ID。文件 ID 包含数据集的划分信息：
  - `training/XXXX`: 表示属于训练集。
  - `test/XXXX`: 表示属于测试集。
  
- **`categories`**: 一个字符串列表，包含文章所属的一个或多个类别标签。
  
- **`text`**: 文章的原始文本内容。

### **2. 数据集分布**
- 总共包含 **10,788** 篇文章。
- 数据集分为训练集和测试集：
  - 训练集: 7,769 篇文章
  - 测试集: 3,019 篇文章
- 类别分布不均衡，热门类别如 `earn` 和 `acq` 占据大多数文章。

### **3. 多标签特性**
- 每篇文章可以属于一个或多个类别。一共有90个类别。
- 示例：
  - 文件 ID: `training/11313`
  - 类别: `['acq', 'earn']`
  - 文本内容: `"General Motors announced a merger with..."`

---

## **常见任务**

### **1. 多标签分类**
- **任务描述**: 给定一篇文章的文本，预测其所属的一个或多个类别标签。
- **挑战**:
  - 类别分布不均衡。
  - 文章可能有多个标签（多标签分类）。

### **2. 单标签分类**
- **任务描述**: 将问题简化为单标签分类任务，仅考虑具有单一标签的文章。
- **示例**:
  - 输入: 文章内容
  - 输出: 单一类别标签（如 `earn`）

### **3. 信息检索**
- **任务描述**: 根据用户输入的关键字，检索相关的新闻文章。

该代码针对的任务是多标签分类任务，一共90个类别
---

## **总结**
- Reuters-21578 数据集是一个经典的多标签文本分类数据集，适用于多种 NLP 任务。
- 数据集通过 NLTK 提供的接口轻松访问，并可转换为 Pandas DataFrame 进行进一步处理。
- 使用以上代码可以轻松生成包含 ID、类别和文本的 DataFrame，方便后续建模与分析。

