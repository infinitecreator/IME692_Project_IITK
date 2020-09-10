# Name - Niraj Kumar
# Roll NO - 160455

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
#class A and class B
#define seed

seed =4
# setting the seed

np.random.seed(seed)

A = np.random.multivariate_normal([1,0],[[1,0],[0,1]],10)
B = np.random.multivariate_normal([0,1],[[1,0],[0,1]],10)


plt.plot(A)
plt.show()
plt.plot(B)
plt.show()
#covariance for generating training and test data

identity_2 = [[1/5,0],[0,1/5]]
#training data variable train_A and corresponding test data
train_A = []
for i in range(100):
    index = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    m_k = A[index]
    train_A.append(np.random.multivariate_normal(m_k,identity_2,1))

train_B = []
for i in range(100):
    index = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    m_k = B[index]
    train_B.append(np.random.multivariate_normal(m_k,identity_2,1))  
    
test_A = []
for i in range(5000):
    index = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    m_k = A[index]
    test_A.append(np.random.multivariate_normal(m_k,identity_2,1))   
test_B = []
for i in range(5000):
    index = np.random.choice([0,1,2,3,4,5,6,7,8,9],p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    m_k = B[index]
    test_B.append(np.random.multivariate_normal(m_k,identity_2,1))

#so now we have the train_A,train_B,test_A,test_B 
#with 100,100,5000,5000 data points respectively    

#taking only the numerical rows from the array type

new_train_A = []
for i in train_A:
    for j in i:
        new_train_A.append(j)
        
#converting the new_train_A data ponints into the dataframe   
df_train_A = pd.DataFrame(new_train_A)
df_train_A.columns = ['train_A_f1','train_A_f2']

#similarly for train_B
new_train_B = []
for i in train_B:
    for j in i:
        new_train_B.append(j)
        
df_train_B = pd.DataFrame(new_train_B)
df_train_B.columns = ['train_B_f1','train_B_f2']   

new_test_A = []
for i in test_A:
    for j in i:
        new_test_A.append(j)
        
df_test_A = pd.DataFrame(new_test_A)    

df_test_A.columns = ['test_A_f1','test_A_f2']          

new_test_B = []
for i in test_B:
    for j in i:
        new_test_B.append(j)   


df_test_B = pd.DataFrame(new_test_B)    

df_test_B.columns = ['test_B_f1','test_B_f2']
df_test_A['label'] = 0
df_test_B['label'] = 1
df_train_A['label'] = 0
df_train_B['label'] = 1

#making copies to the above varibles such that the variables does not get lost

df_test_A.columns = ['f1','f2','Label']
df_test_B.columns = ['f1','f2','Label']
df_train_A.columns = ['f1','f2','Label']
df_train_B.columns = ['f1','f2','Label']
df_testA_copy = df_test_A
test_data = df_testA_copy.append(df_test_B)
train_data = df_train_A.append(df_train_B)

#taking the labels

train_labels = train_data.Label
test_labels = test_data.Label

#our training data
train_data_c = train_data[['f1','f2']]
test_data_c = test_data[['f1','f2']]

from sklearn.metrics import accuracy_score,f1_score

#running the algorithm
acc_vec_train= []
k_points_train = []
for w in range(1,41):
    if (w==0):
        model = KNeighborsClassifier(n_neighbors=1)
        k_points_train.append(1) 
    else:
        model = KNeighborsClassifier(n_neighbors=w)
        k_points_train.append(w)
    model.fit(train_data_c,train_labels) 
    y_pred = model.predict(train_data_c)
    #print(accuracy_score(train_labels,y_pred))
    acc_vec_train.append(accuracy_score(train_labels,y_pred))
    

error_train =[]
for i in acc_vec_train:
    #print(i)
    error_train.append(100-i*100)
    
        
#plotting the training data k_values vs misclassification error
plt.figure(figsize=(12,7))
plt.plot(error_train,color='black', marker='.', linestyle='dotted',linewidth=2, markersize=12)
plt.title('Training Data')
plt.xlabel('k_values')
plt.ylabel('Misclassification_Error_training')
plt.show()  


#for test data

acc_vec_ver2= []
f1_acc = []
k_points_ver2 = []
for w in range(1,41):
    if (w==0):
        model = KNeighborsClassifier(n_neighbors=1)
        k_points_ver2.append(1) 
    else:
        model = KNeighborsClassifier(n_neighbors=w)
        k_points_ver2.append(w)
    model.fit(train_data_c,train_labels) 
    y_pred = model.predict(test_data_c)
    #print(accuracy_score(test_labels,y_pred))
    acc_vec_ver2.append(accuracy_score(test_labels,y_pred))
    f1_acc.append(f1_score(test_labels,y_pred))
    #print("f1_score_values")
    #print(f1_score(test_labels,y_pred))
    #print("\n")

error_vec_ver2 = []
error_f1 =[]
for i in acc_vec_ver2:
    print(i)
    error_vec_ver2.append(100-i*100)  

error_f1 =[]
for i in f1_acc:
    #print(i)
    error_f1.append(100-i*100) 
    
#plotting the misclassification rate with test data accuracy

plt.figure(figsize=(12,7))
plt.plot(error_vec_ver2,color='m', marker='.', linestyle='dotted',linewidth=2, markersize=12)
plt.title('Accuracy_score (Test_Data)')
plt.xlabel('k_values')
plt.ylabel('Misclassification_Error')
plt.show()  

#combining the plots

#combining all_plots
plt.figure(figsize=(12,7))
plt.plot(error_train,color='black', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Training Data")
#plt.figure(figsize=(12,7))
plt.plot(error_vec_ver2,color='red', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Test Data")
plt.title('Accuracy_score (Test_Data & Training_Data)')
plt.xlabel('k_values')
plt.ylabel('Misclassification_Error')
plt.legend()
plt.show()      

#in bigger version

#combining all_plots
plt.figure(figsize=(24,14))
plt.plot(error_train,color='black', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Training Data")
#plt.figure(figsize=(12,7))
plt.plot(error_vec_ver2,color='green', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Test Data")
plt.title('Accuracy_score (Test_Data & Training_Data)')
plt.xlabel('k_values')
plt.ylabel('Misclassification_Error')
plt.legend()
plt.show() 


#plotting in ratios

k_points_train_2 = [1/i for i in k_points_train]
#plotting in ration

#combining all_plots
plt.figure(figsize=(24,14))
plt.plot(k_points_train_2,error_train,color='black', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Training Data")
#plt.figure(figsize=(12,7))
plt.plot(k_points_train_2,error_vec_ver2,color='red', marker='.', linestyle='dotted',linewidth=2, markersize=12,label = "Test Data")
plt.title('Accuracy_score (Test_Data & Training_Data)')
plt.xlabel('k_values')
plt.ylabel('Misclassification_Error')
plt.legend()
plt.show()


#plotting the regression line in misclassification rate
plt.figure(figsize=(14,7))
X1 = k_points_ver2[:]
Y1 = error_train[:]
X2 = k_points_ver2[:]
Y2 = error_vec_ver2[:]

# solve for a and b
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b



#A_A,B_B = best_fit(X1,Y1)

plt.figure(figsize=(12,6))
a, b = best_fit(X1, Y1)

plt.scatter(X1, Y1)
yfit1 = [a + b * xi for xi in X1]
plt.plot(X1, yfit1,'r',label = 'Training')


# help(plt.scatter)
a, b = best_fit(X2, Y2)

plt.scatter(X2, Y2)
yfit2 = [a + b * xi for xi in X2]
plt.plot(X2, yfit2,'m',label = 'test')
plt.legend()
plt.show()


#now using the gaussian Naives bayes classifier

# Using the bayes classification
from sklearn.naive_bayes import GaussianNB
mo = GaussianNB()
mo.fit(train_data_c,train_labels)
y_pr= mo.predict(test_data_c)
print("naive_bayes error is:")
print(100-accuracy_score(test_labels,y_pr)*100)
tr = mo.predict(train_data_c)
print("naive_bayes error on testing data is:")
print(100-accuracy_score(train_labels,tr)*100)

print("The best possible accuracy on the test data form kNN that I got is 77% at the k value = 5")

     
    
       




