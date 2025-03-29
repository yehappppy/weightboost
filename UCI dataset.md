# 一、 UCI 数据集

一共8 个数据集：Ionosphere, German, Pima Indians Diabetes, Breast Cancer (Diagnostic), wpbc, wdbc, Contraceptive, Spambase (全部都是2分类数据集)

## Requirement: Install the ucimlrepo package 
```
pip install ucimlrepo
```


## 1. Ionosphere


### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X = ionosphere.data.features 
y = ionosphere.data.targets 
  
# metadata 
print(ionosphere.metadata) 
  
# variable information 
print(ionosphere.variables) 
```

## 2. German
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# metadata 
print(statlog_german_credit_data.metadata) 
  
# variable information 
print(statlog_german_credit_data.variables) 

```

## 3. Pima Indians Diabetes
### Import the dataset into your python code 
download from kaggle: [Pima Indians Diabetes](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)


## 4. Breast Cancer (Diagnostic)
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_original.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_original.variables) 

```



## 5. wpbc
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_prognostic.data.features 
y = breast_cancer_wisconsin_prognostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_prognostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_prognostic.variables) 

```


## 6. wdbc
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

```


## 7. Contraceptive
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
contraceptive_method_choice = fetch_ucirepo(id=30) 
  
# data (as pandas dataframes) 
X = contraceptive_method_choice.data.features 
y = contraceptive_method_choice.data.targets 
  
# metadata 
print(contraceptive_method_choice.metadata) 
  
# variable information 
print(contraceptive_method_choice.variables) 
```


## 8. Spambase
### Import the dataset into your python code 
```
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 
  
# metadata 
print(spambase.metadata) 
  
# variable information 
print(spambase.variables) 
```