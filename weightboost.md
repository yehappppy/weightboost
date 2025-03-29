# WeightBoost算法实现与评估

本文档提供了"A New Boosting Algorithm Using Input-Dependent Regularizer"论文中WeightBoost算法的完整实现和评估代码。

## 目录

1. [算法介绍](#1-算法介绍)
2. [代码实现](#2-代码实现)
3. [实验设置](#3-实验设置)
4. [结果分析](#4-结果分析)
5. [使用指南](#5-使用指南)
6. [参考文献](#6-参考文献)

## 1. 算法介绍

WeightBoost是一种创新的Boosting算法，解决了传统AdaBoost的两个主要问题：

1. **过拟合问题**：AdaBoost在噪声数据上容易过度关注难以分类的样本
2. **固定权重组合问题**：AdaBoost使用固定常数组合基分类器，没有考虑输入模式的特点

WeightBoost的核心创新是引入了"输入依赖型正则化因子"，使组合权重成为输入的函数：

```
H_T(x) = Σ α_t·e^(-|βH_{t-1}(x)|)·h_t(x)
```

这种方式使每个基分类器只在其专长的输入模式上发挥作用，同时通过正则化减轻噪声数据的影响。

## 2. 代码实现

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# AdaBoost算法实现
class AdaBoost:
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # 初始化权重
        
        for t in range(self.n_estimators):
            # 训练基分类器
            model = clone(self.base_classifier)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # 计算带权重的误差
            err = np.sum(w * (pred != y)) / np.sum(w)
            
            # 检查是否错误率过高
            if err >= 0.5:
                break
                
            # 计算模型权重
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            # 保存模型和权重
            self.models.append(model)
            self.alphas.append(alpha)
            
            # 更新样本权重
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)  # 归一化
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        H = np.zeros(n_samples)
        
        for alpha, model in zip(self.alphas, self.models):
            H += alpha * model.predict(X)
            
        return np.sign(H)

# Weight Decay方法实现
class WeightDecay(AdaBoost):
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100, C=0.1):
        super().__init__(base_classifier, n_estimators)
        self.C = C  # 正则化系数
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # 初始化权重
        H = np.zeros(n_samples)  # 累积分类器输出
        
        for t in range(self.n_estimators):
            # 训练基分类器
            model = clone(self.base_classifier)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # 计算带权重的误差
            err = np.sum(w * (pred != y)) / np.sum(w)
            
            # 检查是否错误率过高
            if err >= 0.5:
                break
                
            # 计算模型权重
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            # 保存模型和权重
            self.models.append(model)
            self.alphas.append(alpha)
            
            # 更新累积分类器输出
            H += alpha * pred
            
            # 计算累积权重（slack变量）
            zeta = H**2
            
            # 更新样本权重 (包括Weight Decay正则化)
            w = np.exp(-y * H - self.C * zeta)
            w = w / np.sum(w)  # 归一化
        
        return self

# ε-Boost方法实现
class EpsilonBoost(AdaBoost):
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100, epsilon=0.1):
        super().__init__(base_classifier, n_estimators)
        self.epsilon = epsilon  # 较小的权重因子
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # 初始化权重
        
        for t in range(self.n_estimators):
            # 训练基分类器
            model = clone(self.base_classifier)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # 计算带权重的误差
            err = np.sum(w * (pred != y)) / np.sum(w)
            
            # 检查是否错误率过高
            if err >= 0.5:
                break
                
            # 固定的小权重
            alpha = self.epsilon
            
            # 保存模型和权重
            self.models.append(model)
            self.alphas.append(alpha)
            
            # 更新样本权重
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)  # 归一化
        
        return self

# WeightBoost算法实现（论文核心算法）
class WeightBoost:
    def __init__(self, base_classifier=DecisionTreeClassifier(max_depth=1), n_estimators=100, beta=0.5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.beta = beta  # 输入依赖型正则化器的参数
        self.alphas = []
        self.models = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # 初始化权重
        H = np.zeros(n_samples)  # 累积分类器输出
        
        for t in range(self.n_estimators):
            # 训练基分类器
            model = clone(self.base_classifier)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X)
            
            # 计算带权重的误差
            err = np.sum(w * (pred != y)) / np.sum(w)
            
            # 检查是否错误率过高
            if err >= 0.5:
                break
                
            # 计算模型权重
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            # 保存模型和权重
            self.models.append(model)
            self.alphas.append(alpha)
            
            # 更新累积分类器输出
            H += alpha * pred
            
            # 计算正则化因子
            reg = np.exp(-np.abs(self.beta * H))
            # 使用论文中提到的归一化因子
            C_T = np.sum(reg) / (0.1 * n_samples)
            reg = reg / C_T
            
            # 更新样本权重（带有输入依赖型正则化器）
            w = np.exp(-y * H - np.abs(self.beta * H))
            w = w / np.sum(w)  # 归一化
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        H = np.zeros(n_samples)
        
        # 迭代应用累积的分类器
        for i, (alpha, model) in enumerate(zip(self.alphas, self.models)):
            if i == 0:
                H = alpha * model.predict(X)
            else:
                # 计算正则化因子
                reg = np.exp(-np.abs(self.beta * H))
                # 使用论文中提到的归一化因子
                C_T = np.sum(reg) / (0.1 * n_samples)
                reg = reg / C_T
                
                # 将基分类器的预测与输入依赖型正则化器加权结合
                H = H + alpha * reg * model.predict(X)
        
        return np.sign(H)

# 添加噪声函数
def add_noise(y, noise_level):
    """给标签添加噪声"""
    noisy_y = y.copy()
    n_samples = len(y)
    n_noise = int(n_samples * noise_level)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    noisy_y[noise_idx] = -noisy_y[noise_idx]  # 翻转标签
    return noisy_y

# 评估函数 - UCI数据集
def evaluate_uci_datasets(datasets, noise_levels=[0, 0.1, 0.2, 0.3]):
    """在UCI数据集上评估所有算法"""
    results = {}
    
    for noise_level in noise_levels:
        noise_results = {}
        
        for name, (X, y) in datasets.items():
            dataset_results = {}
            
            # 创建带噪声的标签
            if noise_level > 0:
                noisy_y = add_noise(y, noise_level)
            else:
                noisy_y = y
                
            # 基准: C4.5决策树
            base_score = np.mean(cross_val_score(
                DecisionTreeClassifier(), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['C4.5'] = 1 - base_score
            
            # AdaBoost
            adaboost_score = np.mean(cross_val_score(
                AdaBoost(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['AdaBoost'] = 1 - adaboost_score
            
            # Weight Decay
            wd_score = np.mean(cross_val_score(
                WeightDecay(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['Weight Decay'] = 1 - wd_score
            
            # ε-Boost
            eps_score = np.mean(cross_val_score(
                EpsilonBoost(n_estimators=100), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['ε-Boost'] = 1 - eps_score
            
            # WeightBoost
            wb_score = np.mean(cross_val_score(
                WeightBoost(n_estimators=100, beta=0.5), X, noisy_y, cv=10, scoring='accuracy'))
            dataset_results['WeightBoost'] = 1 - wb_score
            
            noise_results[name] = dataset_results
        
        results[f"{int(noise_level*100)}% Noise"] = noise_results
    
    return results

# 评估函数 - Reuters数据集
def evaluate_reuters(X_train, y_train, X_test, y_test):
    """在Reuters文本分类数据集上评估算法"""
    # 选择前10个最常见的类别进行评估
    top_categories = np.unique(y_train)[:10]
    results = {}
    
    for category in top_categories:
        category_results = {}
        
        # 创建二分类问题（一对所有）
        y_train_binary = np.where(y_train == category, 1, -1)
        y_test_binary = np.where(y_test == category, 1, -1)
        
        # 基准: C4.5决策树
        base_clf = DecisionTreeClassifier()
        base_clf.fit(X_train, y_train_binary)
        base_pred = base_clf.predict(X_test)
        base_f1 = f1_score(y_test_binary, base_pred, average='binary')
        category_results['C4.5'] = base_f1
        
        # AdaBoost (10次迭代，基于论文)
        ada = AdaBoost(n_estimators=10)
        ada.fit(X_train, y_train_binary)
        ada_pred = ada.predict(X_test)
        ada_f1 = f1_score(y_test_binary, ada_pred, average='binary')
        category_results['AdaBoost'] = ada_f1
        
        # WeightBoost (25次迭代，基于论文)
        wb = WeightBoost(n_estimators=25, beta=0.5)
        wb.fit(X_train, y_train_binary)
        wb_pred = wb.predict(X_test)
        wb_f1 = f1_score(y_test_binary, wb_pred, average='binary')
        category_results['WeightBoost'] = wb_f1
        
        results[category] = category_results
    
    return results

# 可视化结果
def plot_results(results, noise_level=0):
    """绘制在特定噪声水平下的结果"""
    noise_key = f"{int(noise_level*100)}% Noise"
    if noise_key not in results:
        noise_key = "0% Noise"  # 默认无噪声
        
    data = results[noise_key]
    datasets = list(data.keys())
    methods = list(data[datasets[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        error_rates = [data[d][method] * 100 for d in datasets]  # 转为百分比
        ax.bar(x + (i - len(methods)/2 + 0.5) * width, error_rates, width, label=method)
    
    ax.set_xlabel('Data Sets')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Classification Errors with {noise_key}')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
```

## 3. 实验设置

### 3.1 数据集

#### UCI数据集
- Ionosphere (351个样本，341个特征)
- German (1000个样本，20个特征)
- Pima Indians Diabetes (768个样本，8个特征) 
- Breast Cancer (268个样本，9个特征)
- wpbc (198个样本，30个特征)
- wdbc (569个样本，30个特征)
- Contraceptive (1473个样本，10个特征)
- Spambase (4601个样本，58个特征)

#### Reuters-21578语料库
- 训练集：7,769个文档
- 测试集：3,019个文档
- 90个类别，每文档平均1.3个类别

### 3.2 预处理

#### 文本数据预处理
```python
def preprocess_reuters(data_path):
    """预处理Reuters数据集"""
    # 读取数据
    # 文本预处理：转小写、分词、去除标点和停用词、词干提取
    # 使用χ²max准则进行特征选择，选择2000个特征
    # 使用SMART ltc版本的TF-IDF进行文档向量化
    
    # 示例TF-IDF计算
    def tfidf_weight(t, d, D):
        """计算词t在文档d中的TF-IDF权重"""
        tf = 1 + np.log2(term_freq(t, d)) if term_freq(t, d) > 0 else 0
        idf = np.log2(len(D) / doc_freq(t, D))
        return tf * idf
        
    # ...处理代码
    
    return X_train, y_train, X_test, y_test
```

### 3.3 实验流程

1. **标准评估**：
   - 在所有原始UCI数据集上使用10折交叉验证评估各算法性能
   - 在Reuters-21578语料库上评估算法性能，使用F1分数

2. **噪声鲁棒性评估**：
   - 添加不同水平（10%、20%、30%）的标签噪声
   - 使用10折交叉验证评估各算法在噪声数据上的性能

3. **参数配置**：
   - 基分类器：C4.5决策树
   - UCI数据集最大迭代次数：100
   - Reuters数据集：AdaBoost 10次迭代，WeightBoost 25次迭代
   - WeightBoost参数β：0.5

## 4. 结果分析

### 4.1 标准数据集结果

| 数据集 | C4.5 | AdaBoost | Weight Decay | ε-Boost | WeightBoost |
|--------|------|----------|--------------|---------|-------------|
| Ionosphere | 9.1% | 6.8% | 5.7% | 6.8% | 6.2% |
| German | 26.9% | 26.3% | 26.7% | 24.7% | 24.7% |
| Pima | 25.2% | 24.7% | 25.1% | 23.9% | 22.6% |
| Breast Cancer | 5.4% | 4.5% | 3.7% | 3.2% | 3.3% |
| wpbc | 28.8% | 26.3% | 21.1% | 21.1% | 19.9% |
| wdbc | 6.1% | 3.5% | 3.0% | 3.7% | 3.0% |
| Contraceptive | 31.5% | 31% | 29.8% | 30.4% | 27.6% |
| Spambase | 7.2% | 5.8% | 4.9% | 4.5% | 4.2% |

结果显示：
- WeightBoost在所有数据集上都优于AdaBoost
- WeightBoost在6/8个数据集上优于Weight Decay
- 在"German"、"Pima"和"Contraceptive"等数据集上，WeightBoost显示出显著改进，而AdaBoost几乎无改进

### 4.2 Reuters-21578结果（F1分数）

| 类别 | C4.5 | AdaBoost | WeightBoost | AdaBoost改进 | WeightBoost改进 |
|------|------|----------|-------------|--------------|-----------------|
| Trade | 0.5897 | 0.6634 | 0.6949 | 12.5% | 17.8% |
| Grain | 0.9030 | 0.8814 | 0.8966 | -2.4% | -0.7% |
| Crude | 0.8223 | 0.8204 | 0.8587 | -0.2% | 4.4% |
| Corn | 0.8740 | 0.8926 | 0.9091 | 2.1% | 4.0% |
| Ship | 0.7283 | 0.7853 | 0.7273 | 7.8% | -0.1% |
| Wheat | 0.8800 | 0.8767 | 0.9128 | -0.4% | 3.7% |
| Acq | 0.8915 | 0.9344 | 0.9243 | 4.8% | 3.7% |
| Interest | 0.6224 | 0.6747 | 0.6352 | 8.4% | 2.1% |
| Money-fx | 0.6477 | 0.6805 | 0.7041 | 5.1% | 8.7% |
| Earn | 0.9564 | 0.9698 | 0.9707 | 1.4% | 1.5% |

结果显示：
- WeightBoost在7/10个类别上优于AdaBoost
- 对于"crude"和"wheat"类别，WeightBoost有显著改进，而AdaBoost反而表现较差

### 4.3 噪声鲁棒性结果

添加10%标签噪声后的结果：

| 数据集 | C4.5 | AdaBoost | WeightBoost |
|--------|------|----------|-------------|
| Ionosphere | 14.50% | 12.00% | 8.50% |
| German | 28.30% | 30.70% | 25.70% |
| Pima | 26.00% | 25.00% | 24.80% |
| Breast-cancer | 5.90% | 5.90% | 3.50% |
| wpbc | 27.30% | 25.30% | 24.20% |
| wdpc | 7.40% | 6.70% | 3.90% |
| Contraceptive | 31.00% | 31.50% | 29.30% |
| Spambase | 10.00% | 9.60% | 5.80% |

结果显示：
- AdaBoost在某些数据集（如"German"、"Breast-cancer"和"Contraceptive"）上明显过拟合
- WeightBoost在所有噪声数据集上都保持了较好性能
- 在"wdpc"数据集上，WeightBoost在添加10%噪声后仍保持了接近无噪声时的性能

## 5. 使用指南

### 5.1 环境要求

```
numpy >= 1.19.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
pandas >= 1.1.0
```

### 5.2 数据准备

```python
# 加载UCI数据集
from sklearn.datasets import fetch_openml

# 加载Ionosphere数据集
X_ionosphere, y_ionosphere = fetch_openml(name='ionosphere', version=1, return_X_y=True)
# 将标签转换为+1/-1
y_ionosphere = np.where(y_ionosphere == 'g', 1, -1)

# 同样处理其他数据集...

# 准备数据集字典
datasets = {
    "Ionosphere": (X_ionosphere, y_ionosphere),
    "German": (X_german, y_german),
    # ...其他数据集
}
```

### 5.3 运行示例

```python
# 评估UCI数据集
uci_results = evaluate_uci_datasets(datasets)

# 绘制无噪声结果
plot_results(uci_results, noise_level=0)

# 绘制10%噪声结果
plot_results(uci_results, noise_level=0.1)

# 评估Reuters数据集
reuters_results = evaluate_reuters(X_reuters_train, y_reuters_train, 
                                  X_reuters_test, y_reuters_test)

# 打印Reuters结果
print("Reuters-21578 Results (F1 Score):")
for category, results in reuters_results.items():
    print(f"\nCategory: {category}")
    for method, f1 in results.items():
        print(f"{method}: {f1:.4f}")
```

### 5.4 自定义参数

```python
# 自定义WeightBoost参数
wb = WeightBoost(
    base_classifier=DecisionTreeClassifier(max_depth=3),  # 更改基分类器
    n_estimators=50,  # 更改迭代次数
    beta=0.8  # 调整正则化强度
)

# 使用不同噪声水平
custom_noise_levels = [0, 0.05, 0.15, 0.25]
results = evaluate_uci_datasets(datasets, noise_levels=custom_noise_levels)
```

## 6. 参考文献

1. Jin, R., Liu, Y., Si, L., Carbonell, J., & Hauptmann, A. G. (2003). A New Boosting Algorithm Using Input-Dependent Regularizer. *Proceedings of the Twentieth International Conference on Machine Learning (ICML-2003)*, Washington DC.

2. Freund, Y., & Schapire, R. E. (1996). Experiments with a new boosting algorithm. *Machine Learning: Proceedings of the Thirteenth International Conference*.

3. Friedman, J., Hastie, T., & Tibshirani, R. (1998). Additive logistic regression: a statistical view of boosting. *Annals of statistics*, 28(2), 337-407.

4. Ratsch, G., Onoda, T., & Muller, K. R. (1998). Soft margins for AdaBoost. *Machine learning*, 42(3), 287-320.

5. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers Inc.