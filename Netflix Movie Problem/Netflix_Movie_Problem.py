# -*- coding:utf-8 -*-
import os
import os.path
import numpy as np
import pandas as pd
import re

# 推荐算法：
# 基于临域
# 隐语义模型与矩阵分解，不考虑评分时间对评分带来的影响
# 平均值
# 神经网络

# 数据集：
# 480,189名用户对17,770部

MOVIE_NUM = 17770
USER_NUM = 480189

"""
f = open(r'combined_data_1.txt','r')
f_1 = open(r'combined_data_label_1.txt','w')
label="-1"
for i in f:
    if(re.findall(':\n\Z',i)):
      label=i.split(':')[0]
    else:
      if(int(label)>0):
        i=re.sub('\n',',',i)+label+'\n'
        f_1.write(i)
        print(i)
f.close()

f = open(r'combined_data_2.txt','r')
f_1 = open(r'combined_data_label_2.txt','w')
label="-1"
for i in f:
    if(re.findall(':\n\Z',i)):
      label=i.split(':')[0]
    else:
      if(int(label)>0):
        i=re.sub('\n',',',i)+label+'\n'
        f_1.write(i)
        print(i)
f.close()

f = open(r'combined_data_3.txt','r')
f_1 = open(r'combined_data_label_3.txt','w')
label="-1"
for i in f:
    if(re.findall(':\n\Z',i)):
      label=i.split(':')[0]
    else:
      if(int(label)>0):
        i=re.sub('\n',',',i)+label+'\n'
        f_1.write(i)
        print(i)
f.close()

f = open(r'combined_data_4.txt','r')
f_1 = open(r'combined_data_label_4.txt','w')
label="-1"
for i in f:
    if(re.findall(':\n\Z',i)):
      label=i.split(':')[0]
    else:
      if(int(label)>0):
        i=re.sub('\n',',',i)+label+'\n'
        f_1.write(i)
        print(i)
f.close()
"""

# CD_1 = pd.read_csv('combined_data_label_1.txt', sep=',', header=None).values

User_IDs = np.load('user_ids.npy') # 用户RawID（顺序）

print("----type----")
print(type(User_IDs))
print("----shape----")
print(User_IDs.shape)
print("----data----")
print(User_IDs[0:3])
print(User_IDs[-1])

Oberved_ratings = np.load('observed_ratings.npy') # 电影ID 用户RawID 用户评分

print("----type----")
print(type(Oberved_ratings))
print("----shape----")
print(Oberved_ratings.shape)
print("----data----")
print(Oberved_ratings)
print(Oberved_ratings[:,0])

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cupy as cp
import numba 
from numba import jit
import csv
# from scipy.sparse import coo_matrix

User_Raw_IDs = np.load('user_ids.npy')
User_Raw_IDs_df = pd.DataFrame({'User_Raw_ID':User_Raw_IDs})
User_IDs_Comparison_Table=pd.DataFrame({'User_ID':User_Raw_IDs_df.index.values,'User_Raw_ID':User_Raw_IDs})
print(User_IDs_Comparison_Table)

# 存在特殊符号乱码，如果使用pandas库进行读取，需要使用ISO-8859-1编码方式进行读取。
# 存在第四、五、六列数据，不能使用pandas库进行读取，需要使用csv库重新合并为新的csv文件。
# 电影年份存在NULL，以所有电影的最早年份进行填充。
# 电影ID为'6484'的电影数据，处于电影ID为'6483'的电影的电影名中，通过Excel进行手动更改。

column1=[]
column2=[]
column3=[]
last_movie_id=0

"""
with open('movie_titles.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        column1.append(row[0])
        if int(last_movie_id)+1!=int(row[0]):
            print(row[0]) # 找出电影ID的不连续处。
        last_movie_id=row[0]
        if row[1]!=None:
          column2.append(row[1])
        else:
          column2.append('1909') # 空缺的电影视为最早的年份1909出版
        column3.append(row[2]+row[3]+row[4]+row[5])

Movie_Titles_df = pd.DataFrame({'Movie_RAW_ID':column1})
Movie_Titles_Comparison_Table = pd.DataFrame({'Movie_ID':list(Movie_Titles_df.index.values),'Movie_RAW_ID':column1,'Movie_Name':column3,'Year':column2})
"""
# Movie_Titles_Comparison_Table.to_csv('movie_titles_processed.csv',index=False,header=True)
Movie_Titles_Comparison_Table=pd.read_csv('movie_titles_processed.csv',encoding='ISO-8859-1')
print(Movie_Titles_Comparison_Table)

"""
User_IDs_Comparison_Table_dict=dict(zip(User_IDs_Comparison_Table['User_Raw_ID'],User_IDs_Comparison_Table['User_ID']))
Movie_IDs_Comparison_Table_dict=dict(zip(Movie_Titles_Comparison_Table['Movie_RAW_ID'],Movie_Titles_Comparison_Table['Movie_ID']))

Rating_All = np.load('observed_ratings.npy')
User_IDs_for_Rating_All=[User_IDs_Comparison_Table_dict[i] for i in Rating_All[:,1]]
Movie_IDs_for_Rating_All=[Movie_IDs_Comparison_Table_dict[i] for i in Rating_All[:,0]] # 生成字典时有可能使键转变为整数或字符串，因此有可能需要str()。
Rating_All_df=pd.DataFrame({'Movie_ID':Movie_IDs_for_Rating_All,'Movie_Raw_ID':Rating_All[:,0],'User_Raw_ID':Rating_All[:,1],'User_ID':User_IDs_for_Rating_All,'User_Rating':Rating_All[:,2]})
"""

# Rating_All_df.to_csv('User_Movie_Rating.csv',index=False,header=True)
Rating_All_df=pd.read_csv('User_Movie_Rating.csv',encoding='ISO-8859-1')
print(Rating_All_df)

# 获取数据
ratings_df=pd.DataFrame({'movieRow':Rating_All_df['Movie_ID'],'userId':Rating_All_df['User_ID'],'rating':Rating_All_df['User_Rating']})
movies_df=pd.DataFrame({'movieRow':Movie_Titles_Comparison_Table['Movie_ID'],'movieID':Movie_Titles_Comparison_Table['Movie_RAW_ID'],'Movie_Year':Movie_Titles_Comparison_Table['Year'],'title':Movie_Titles_Comparison_Table['Movie_Name']})

# ratings_df = pd.read_csv('real_ratings.csv')
# movies_df = pd.read_csv('movies.csv')
print(ratings_df) # userId movieRow（movieID） rating
print(movies_df) # movieRow（movieID） movieID（Raw）  title

userNo = max(ratings_df['userId'])+1
movieNo = max(ratings_df['movieRow'])+1
print(userNo)
print(movieNo)

# 创建电影评分表
"""
rating = np.zeros((userNo,movieNo),dtype='float16') # numpy通过扩大虚拟内存来容纳，设置类型为float32，大小需要大概32GiB。
# rating = cp.asarray(rating)
print("finish step 1")

for index,row in ratings_df.iterrows():
# iterrows逐行遍历。
# 行索引，行值
    rating[int(row['userId']),int(row['movieRow'])]=row['rating'] # 将评分进行记录，[用户ID，电影ID]。
"""

# np.save('rating_matrix_1.npy',rating)
rating=np.load('Rating_Matrix.npy').astype('float16')
print(rating)
print("finish step 2")

"""
def recommend(userID,lr,alpha,d,n_iter,data):
    '''
    userID(int):推荐用户ID
    lr(float):学习率
    alpha(float):权重衰减系数
    d(int):矩阵分解因子(即元素个数)
    n_iter(int):训练轮数
    data(ndarray):用户-电影评分矩阵
    ''' 
    # 获取用户数m与电影数n
    m,n = data.shape 
    # 初始化参数矩阵（隐变量矩阵）
    # x = np.random.uniform(0,1,(m,d))
    # w = np.random.uniform(0,1,(d,n))
    x = cp.random.uniform(0,1,(m,d)).astype('float16')
    w = cp.random.uniform(0,1,(d,n)).astype('float16')
    print("finish step 3")
    # 创建评分记录表，无评分记为0，已有评分记为1
    # record = np.array(data>0,dtype='int8')
    record = np.array(data>0,dtype='int8')
    print("finish step 4")
    # 进行n_iter次迭代的梯度下降，更新参数           
    for i in range(n_iter):
        print("iter_begin")
        # x_grads = np.dot(np.multiply(record,np.dot(x,w)-data),w.T)
        x_grads = cp.dot(cp.multiply(record,cp.dot(x,w)-data),w.T)
        # np.multiply为对应位置乘积，np.dot为矩阵乘法，.T为转置。
        # w_grads = np.dot(x.T,np.multiply(record,np.dot(x,w)-data))
        w_grads = cp.dot(x.T,cp.multiply(record,cp.dot(x,w)-data))
        # 同时进行梯度下降。
        x = alpha*x - lr*x_grads
        w = alpha*w - lr*w_grads
        print("iter finish")
    # 预测，两分解矩阵相乘，获得预测值矩阵。
    # predict = np.dot(x,w)
    predict = cp.dot(x,w)
    print("finish step 5")
    # 将用户未看过的每一部电影分值从低到高进行排列
    for i in range(n):
        if record[userID-1][i] == 1 :
            predict[userID-1][i] = 0 
    recommend = np.argsort(predict[userID-1])
    # recommend = cp.argsort(predict[userID-1])
    a = recommend[-1]
    b = recommend[-2]
    c = recommend[-3]
    d = recommend[-4]
    e = recommend[-5]
    print('为用户%d推荐的电影为：\n1:%s\n2:%s\n3:%s\n4:%s\n5:%s。'\
          %(userID,movies_df['title'][a],movies_df['title'][b],movies_df['title'][c],movies_df['title'][d],movies_df['title'][e]))
    return x,w,predict
"""

@jit
def Grad_Desc(R,K=10,max_iter=20,alpha=0.0001,lamda=0.002):
    M,N = R.shape
    P=cp.random.rand(M,K).astype('float16')
    Q=cp.random.rand(N,K).T.astype('float16')
    # R=cp.asarray(R)
    for step in range(max_iter):
      for u in range(M):
        print(step,u)
        for i in range(N):
          if R[u][i]>0:
            err=cp.dot(P[u,:],Q[:,i])-R[u][i]
            # print('{},{}'.format(u,i))
            for k in range(K):
              P[u][k]=P[u][k]-alpha*(2*err*Q[k][i]+2*lamda*P[u][k])
              Q[k][i]=Q[k][i]-alpha*(2*err*P[u][k]+2*lamda*Q[k][i])
      np.load('P_matrix.npy_'+str(step)+'.npy',cp.asnumpy(P))
      np.load('Q_matrix.npy_'+str(step)+'.npy',cp.asnumpy(Q))
    np.load('P_matrix.npy',cp.asnumpy(P))
    np.load('Q_matrix.npy',cp.asnumpy(Q))
    print("finish step 3")

    cost=0
    for u in range(M):
      for i in range(N):
        if R[u][i]>0:
          cost+=(cp.dot(P[u,:],Q[:,i])-R[u][i])**2
          #加上正则化项
          for k in range(K):
            cost+=lamda*(P[u][k]**2+Q[k][i]**2)
      if cost<0.0001:
        break
    
    np.load('Cost.npy',cost)
    print("finish step 4")

    predict=cp.dot(P,Q)
    np.load('Predict_matrix.npy',cp.asnumpy(predict))
    print("finish step 5")

k=20

# x,w,predict=recommend(666,1e-4,0.999,k,50,rating) 

Grad_Desc(rating,K=10,max_iter=20,alpha=0.0001,lamda=0.002)

"""
为用户666推荐的电影为：
1:Aquamarine (2006)
2:It's a Boy Girl Thing (2006)
3:Kill the Messenger (2014)
4:Onion Field, The (1979)
5:Wind Rises, The (Kaze tachinu) (2013)。
"""
np.save("x.npy",x)
np.save("w.npy",w)
np.save("predict.npy",predict)
# np.save("x.npy",cp.asnumpy(x))
# np.save("w.npy",cp.asnumpy(w))
# np.save("predict.npy",cp.asnumpy(predict))
print("finish step 6")
