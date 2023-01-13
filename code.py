import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



dataset={'train.csv','test.csv'}
for path in dataset:
    data=pd.read_csv(path)
    #cat_columns = data.select_dtypes(include='O').columns  #获取非数字的列
    job = LabelEncoder()        #对非数字的特征进行编码
    data['job'] = job.fit_transform(data['job'])
    data['marital'] = data['marital'].map({'unknown': 0, 'single': 1, 'married': 2, 'divorced': 3})
    data['education'] = data['education'].map({'unknown': 0, 'illiterate': 1, 'basic.4y': 2, 'basic.6y': 3, \
                                           'basic.9y': 4, 'high.school': 5, 'university.degree': 6,
                                           'professional.course': 7})

    data['housing'] = data['housing'].map({'unknown': 0, 'no': 1, 'yes': 2})
    data['loan'] = data['loan'].map({'unknown': 0, 'no': 1, 'yes': 2})
    data['contact'] = data['contact'].map({'cellular': 0, 'telephone': 1})
    data['day_of_week'] = data['day_of_week'].map({'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4})
    data['poutcome'] = data['poutcome'].map({'nonexistent': 0, 'failure': 1, 'success': 2})
    data['default'] = data['default'].map({'unknown': 0, 'no': 1, 'yes': 2})
    data['month'] = data['month'].map({'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, \
                     'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
    if path=='test.csv':
        y_id=data['id']
        test=data.copy()

    if path=='train.csv':
        data['subscribe'] = data['subscribe'].map({'no': 0, 'yes': 1})
        data.drop(['id'], axis=1, inplace=True)
        data.drop(['month'], axis=1, inplace=True)
        data.drop(['day_of_week'], axis=1, inplace=True)
        Y = data['subscribe']
        #print(Y)
        X=data.drop(['subscribe'],axis=1)
        #print(X)
x_train,x_test,y_train,y_test = train_test_split( X, Y,test_size=0.3,random_state=1)  #划分训练集与测试集

def DecisionTree(x_train,y_train):     #决策树
    model = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_impurity_decrease=0.0)
    model.fit(x_train, y_train)
    #score=model.score(x_test,y_test)
    #print(score)
    return model

def Gaussian_NB(x_train,y_train):   #先验为高斯分布，适用于样本特征的分布大部分为连续值
    model=GaussianNB()
    model.fit(x_train,y_train)
    return model

def Multinomial_NB(x_train,y_train): #先验为多项式分布的朴素贝叶斯，适用于样本特征的分布大部分为多元离散
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model


def Bernoulli_NB(x_train,y_train):  #先验为伯努利分布，适用于样本特征是二元离散值或者很稀疏的多元离散值
    model = BernoulliNB()
    model.fit(x_train, y_train)
    return model

def svm_rbf(x_train,y_train):     #svm
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    return model

def svm_linear(x_train,y_train):
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model

def svm_poly(x_train,y_train):
    model = SVC(kernel='poly')
    model.fit(x_train, y_train)
    return model


model=DecisionTree(x_train,y_train)  #0.8779259259259259
#model=Gaussian_NB(x_train,y_train)  #0.8665185185185185
#model=Multinomial_NB(x_train,y_train) #error
#model=Bernoulli_NB(x_train,y_train)  #0.8638518518518519
#model=svm_rbf(x_train,y_train)      #0.8694814814814815
#model=svm_linear(x_train,y_train)   #运行时间过长
#model=svm_poly(x_train,y_train)      #0.8694814814814815

score=model.score(x_test,y_test)
y_label=np.array(y_test)
#决策树 朴素贝叶斯计算y_pre
#y_pre=model.predict_proba((x_test))[:,1]
#SVM计算y_rpe
y_pre=model.decision_function(x_test)
print(y_pre)
print(y_label)
print(score)
fpr, tpr, thersholds = roc_curve(y_label, y_pre)

for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr[i], tpr[i], value))

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

#y_pred = model.predict(test)
#print(y_pred)


