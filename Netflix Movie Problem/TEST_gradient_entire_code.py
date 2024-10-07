# -*- coding: utf-8 -*-
import multiprocessing as mlp
import numpy as np
import cupy as cp
import csv
import numba 
from numba import jit
from scipy.sparse import csc_matrix,find
import pandas as pd
import tensorflow as tf # tensorflow和keras在使用GPU的时候有个特点，就是默认全部占满显存。
from tensorflow import keras # tensorflow == 2.X version
import warnings
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 设置警告屏蔽等级
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 多GPU时使用

"""
因Netflix数据集过大，使用同类型的Movielens-1m数据集进行调试。
"""

# 处理其中一部分数据。

def read_part_data(file_path):
    with open(file_path, 'r') as f:
        movie_user_ratings = []
        lines = f.read().split('\n')
        lines.remove('')
        movieId = 0
        for line in lines:
            if ':' in line:
                movieId = int(line.replace(':',''))
            else:
                info = line.split(',')
                userId, userRating = int(info[0]), int(info[1])
                movie_user_ratings.append((movieId, userId, userRating))
        movie_user_ratings = np.array(movie_user_ratings)
        userIds = np.unique(movie_user_ratings[:,1])
        return movie_user_ratings, userIds

# 处理全部数据

def read_all_data(train_file_list):
    first_file = True
    for file in train_file_list:
        if first_file:
            movie_user_ratings, userIds, rating_counts, rating_sums = read_part_data(file)
            first_file = False
        else:
            ratings, Ids, counts, sums = read_part_data(file)
            movie_user_ratings = np.concatenate([movie_user_ratings, ratings], axis=0) # 拼接数据。
            userIds = np.concatenate([userIds, Ids], axis=0) # 拼接数据。
    userIds = np.unique(userIds) # 用户RawID去除重复。
    np.sort(userIds) # 用户RawID升序排序。
    return movie_user_ratings, userIds

# 将处理后的数据进行保存

def prepare_all_data():
    movie_user_ratings, userIds = read_all_data(TRAIN_FILES)
    np.save('observed_ratings.npy', movie_user_ratings)
    np.save('user_ids.npy', userIds)
    print('Finish writing train data into numpy-array files')

# prepare_all_data()

"""
通过对movie_titles.csv进行预处理，获得和存储电影ID映射表。保存后直接加载即可。
数据预处理：
# movie_titles.csv存在特殊符号乱码，如果使用pandas库进行读取，需要使用ISO-8859-1编码方式进行读取。
# movie_titles.csv存在第四、五、六列数据，不能使用pandas库进行读取，需要使用csv库逐行读取并重新合并为新的csv文件，才能用pandas库进行读取。
# movie_titles.csv电影年份存在NULL，以所有已知电影的最早年份进行填充。
# movie_titles.csv电影ID为'6484'的电影数据，处于电影ID为'6483'的电影的电影名中，只能通过Excel进行手动更改。
"""

"""
column1=[]
column2=[]
column3=[]
last_movie_id=0

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
# print("Movie_Titles_Comparison_Table:")
# print(Movie_Titles_Comparison_Table)

"""
获得和存储用户ID映射表。保存后直接加载即可。
"""

"""
User_Raw_IDs = np.load('user_ids.npy')
User_Raw_IDs_df = pd.DataFrame({'User_Raw_ID':User_Raw_IDs})
User_IDs_Comparison_Table=pd.DataFrame({'User_ID':User_Raw_IDs_df.index.values,'User_Raw_ID':User_Raw_IDs})
"""

# User_IDs_Comparison_Table.to_csv('User_IDs_Comparison_Table.csv',index=False,header=True)
User_IDs_Comparison_Table=pd.read_csv('User_IDs_Comparison_Table.csv',encoding='ISO-8859-1')
# print("User_IDs_Comparison_Table:")
# print(User_IDs_Comparison_Table)

"""
获得和存储用户ID-电影ID－评分表。
"""

User_IDs_Comparison_Table_dict=dict(zip(User_IDs_Comparison_Table['User_Raw_ID'],User_IDs_Comparison_Table['User_ID']))
Movie_IDs_Comparison_Table_dict=dict(zip(Movie_Titles_Comparison_Table['Movie_RAW_ID'],Movie_Titles_Comparison_Table['Movie_ID']))
Movie_Raw_IDs_Names_Comparision_Comparison_Table_dict=dict(zip(Movie_Titles_Comparison_Table['Movie_RAW_ID'],Movie_Titles_Comparison_Table['Movie_Name']))

"""
Rating_All = np.load('observed_ratings.npy')
User_IDs_for_Rating_All=[User_IDs_Comparison_Table_dict[i] for i in Rating_All[:,1]]
Movie_IDs_for_Rating_All=[Movie_IDs_Comparison_Table_dict[i] for i in Rating_All[:,0]] # 生成字典时有可能使键转变为整数或字符串，因此有可能需要str()。
Rating_All_df=pd.DataFrame({'Movie_ID':Movie_IDs_for_Rating_All,'Movie_Raw_ID':Rating_All[:,0],'User_Raw_ID':Rating_All[:,1],'User_ID':User_IDs_for_Rating_All,'User_Rating':Rating_All[:,2]})

# Rating_All_df.to_csv('User_Movie_Rating.csv',index=False,header=True)
Rating_All_df=pd.read_csv('User_Movie_Rating.csv',encoding='ISO-8859-1') # 在Linux服务器编译时，分隔符有可能失效。
print("User_Movie_Rating:")
print(Rating_All_df)
"""

"""
导入数据。
"""

"""
ratings_df=pd.DataFrame({'movieRow':Rating_All_df['Movie_ID'],'userId':Rating_All_df['User_ID'],'rating':Rating_All_df['User_Rating']})
movies_df=pd.DataFrame({'movieRow':Movie_Titles_Comparison_Table['Movie_ID'],'movieID':Movie_Titles_Comparison_Table['Movie_RAW_ID'],'Movie_Year':Movie_Titles_Comparison_Table['Year'],'title':Movie_Titles_Comparison_Table['Movie_Name']})
print("userId movieRow（movieID）rating:")
print(ratings_df)
print("movieRow（movieID）movieID（Raw）title")
print(movies_df)
"""

"""
获得用户总数和电影总数。
"""

"""
userNo = max(ratings_df['userId'])+1
movieNo = max(ratings_df['movieRow'])+1
print("User Number:")
print(userNo)
print("Movie Number:")
print(movieNo)
"""

"""
获得和存储用户-物品矩阵（评分矩阵）。
"""

"""
rating = np.zeros((userNo,movieNo),dtype='float16') # numpy通过扩大虚拟内存来容纳，设置类型为float32，大小需要大概32GiB。

for index,row in ratings_df.iterrows():
# iterrows逐行遍历。
# 行索引，行值
    rating[int(row['userId']),int(row['movieRow'])]=row['rating'] # 将评分进行记录，[用户ID，电影ID]。
"""

"""
# np.save('Rating_Matrix.npy',rating)
rating=np.load('Rating_Matrix.npy')
# rating=np.load('Rating_Matrix_TEST_QUICKLY.npy')
print("Rating Matrix:")
print(rating)
print("finish load Rating Matrix!")
"""

"""
梯度下降，矩阵乘法实现。全部使用cupy加速则显存不足。部分使用cupy加速效率仍太低，不采用。
"""

def recommend(userID,lr,alpha,d,n_iter,data):
    # 获取用户数m与电影数n
    m,n = data.shape 
    # 初始化参数矩阵（隐变量矩阵）
    x = np.random.uniform(0,1,(m,d))
    w = np.random.uniform(0,1,(d,n))
    # x = cp.random.uniform(0,1,(m,d))
    # w = cp.random.uniform(0,1,(d,n))
    # 创建评分记录表，无评分记为0，已有评分记为1
    record = np.array(data>0,dtype='int8')
    # record = cp.array(data>0,dtype=int)
    # 进行n_iter次迭代的梯度下降，更新参数           
    for i in range(n_iter):
        x_grads = np.dot(np.multiply(record,np.dot(x,w)-data),w.T) # np.multiply为对应位置乘积，np.dot为矩阵乘法，.T为转置。
        # x_grads = cp.dot(cp.multiply(record,cp.dot(x,w)-data),w.T)
        w_grads = np.dot(x.T,np.multiply(record,np.dot(x,w)-data))
        # w_grads = cp.dot(x.T,cp.multiply(record,cp.dot(x,w)-data))
        # 同时进行梯度下降。
        x = alpha*x - lr*x_grads
        w = alpha*w - lr*w_grads
        print("iter finish")
    # 预测，两分解矩阵相乘，获得预测值矩阵。
    predict = np.dot(x,w)
    np.save("x.npy",x)
    np.save("w.npy",w)
    np.save("predict_final.npy",predict)
    # predict = cp.dot(x,w)
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


# x,w,predict=recommend(1,1e-4,0.999,20,50,rating) 

"""
梯度下降，循环实现。节省内存，使用numba加速效率仍太低，不采用。
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

    predict=cp.dot(P,Q)
    np.load('Predict_matrix.npy',cp.asnumpy(predict))

# Grad_Desc(rating,K=10,max_iter=20,alpha=0.0001,lamda=0.002)

"""
梯度下降，矩阵乘法实现。带有RMSE计算。使用稀疏矩阵计算效率仍太低，不采用。
"""

def Gradient_Descent_RMSE(Rating_Matrix,K=5,Max_Iter=10,alpha=0.0001,lamda=0.002):
    M,N=Rating_Matrix.shape
    P=np.random.randn(M,K).astype('float16')
    Q=np.random.randn(N,K).astype('float16')
    # Gradient Descent
    Record=np.array(Rating_Matrix>0,dtype='int8')
    Record=csc_matrix(Record)
    # np.save('row_Record.npy',find(Record)[0])
    # np.save('col_Record.npy',find(Record)[1])
    # np.save('data_Record.npy',find(Record)[2])
    # Record=csc_matrix((np.load('data_Record.npy',(np.load('row_Record.npy'),np.load('col_Record.npy')))))
    Rating_Matrix=csc_matrix(Rating_Matrix)
    # np.save('row_Rating_Matrix.npy',find(Rating_Matrix)[0])
    # np.save('col_Rating_Matrix.npy',find(Rating_Matrix)[1])
    # np.save('data_Rating_Matrix.npy',find(Rating_Matrix)[2])
    # Rating_Matrix=csc_matrix((np.load('data_Rating_Matrix.npy',(np.load('row_Rating_Matrix.npy'),np.load('col_Rating_Matrix.npy')))))
    Number_of_None_Zero=find(Record)[2]
    # J=np.zeros(Max_Iter)
    RMSE=np.zeros(Max_Iter)
    for step in range(Max_Iter):
        print("Iter {} begin !".format(step))
        P-=alpha * (np.dot(np.multiply(Record,(np.dot(P,Q.T)-Rating_Matrix)),Q) + 2 * lamda * P)
        print("Get P!")
        Q-=alpha * (np.dot(np.multiply(Record,(np.dot(P,Q.T)-Rating_Matrix)),P) + 2 * lamda * Q)
        print("Get Q!")
        # 平方差损失函数：
        # J[step] = 1/2*np.sum(np.sum(np.square(np.multiply(Record, (Rating_Matrix - np.dot(P, Q.T)))))) + lamda * np.sum(np.sum(np.square(P))) + lamda * np.sum(np.sum(np.square(Q)))
        # 均方根差RMSE计算：
        RMSE[step] = np.sqrt( np.sum( np.square( np.multiply( Record,(np.dot(P,Q.T)-Rating_Matrix) ) ) ) / Number_of_None_Zero )
        print("Get RMSE!")
        print("Iter {} finish !".format(step))
    return P,Q,RMSE

# P,Q,RMSE=Gradient_Descent_RMSE(rating,K=5,Max_Iter=10,alpha=0.0001,lamda=0.002)

"""
梯度下降，矩阵乘法（深度学习框架）实现。带有RMSE计算。调用GPU性能，效率高，采用。
保存模型后，直接加载即可。
"""

def Train():
    # ------ 读入数据 ------ #
    
    # dataset = pd.read_csv("ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"])
    dataset=pd.read_csv('User_Movie_Rating.csv',encoding='ISO-8859-1')
    dataset.sort_values(by=['User_ID','Movie_ID'],inplace=True)
    print(dataset)
    # 数据预处理，下标从0开始，去除缺失值使得值连续
    # dataset.user_id = dataset.user_id.astype('category').cat.codes.values
    # dataset.item_id = dataset.item_id.astype('category').cat.codes.values
    dataset.User_ID = dataset.User_ID.astype('category').cat.codes.values
    dataset.Movie_ID = dataset.Movie_ID.astype('category').cat.codes.values
    print(dataset)
    # 获取用户和项目列表
    # user_arr = dataset.user_id.unique()
    # movies_arr = dataset.item_id.unique()
    user_arr = dataset.User_ID.unique()
    movies_arr = dataset.Movie_ID.unique()
    print(user_arr)
    print(movies_arr)
    # 获取用户和项目数量
    n_users, n_movies = len(user_arr), len(movies_arr)
    # M1:6040 3706
    # Netflix:480189 17770
    print(n_users,n_movies)
    
    # ------ 设置Keras参数 ------ #

    # 设置隐变量个数
    n_latent_factors = 20 # K
    
    # 设置项目参数
    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding = keras.layers.Embedding(n_movies, n_latent_factors, name='Movie-Embedding')(movie_input)
    movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
    
    # 设置用户参数
    user_input = keras.layers.Input(shape=[1], name='User')
    user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='User-Embedding')(user_input)
    user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
    
    # 计算项目向量与用户张量的点乘
    prod = keras.layers.dot([movie_vec, user_vec], axes=1, name='DotProduct')
    
    # 创建用户-项目模型
    model = keras.Model([user_input, movie_input], prod)
    
    # 设置模型优化器、损失函数、测量指标
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae','mse',tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    # mes就是均方差。
    # tf.keras.metrics.RootMeanSquaredError(name='rmse')为根均方差。
    
    # ------ 训练模型 ------ #
    
    # 训练用户-项目模型
    # verbose=0：不输出日志；verbose=1：输出每一个step的训练进度及日志（epoch中每个batch的记录）；verbose=2：输出每个epochs的日志
    # hist = model.fit([dataset.user_id, dataset.item_id], dataset.rating, epochs=1, verbose=2)
    hist = model.fit([dataset.User_ID, dataset.Movie_ID], dataset.User_Rating, steps_per_epoch=1000,epochs=5, verbose=1,workers=6)
    # steps_per_epoch等于总数据大小除以批次大小（batch_size），其实就是iteration，等同于设置batch_size。
    # 对batch_size进行设置不会影响最终训练结果，但是会占用更多的内存和显存，当显存不够时，深度学习框架会自动使用共享GPU内存。
    model.save('Keras_Model_Netflix_epoch',save_format='tf')
    
    loss=hist.history['loss']
    np_loss=np.array(loss)
    np.save('loss',np_loss)

    hist_df=pd.DataFrame(hist.history)
    print(hist_df)
    hist_df['epoch']=hist.epoch
    print(hist_df)
    hist_df.to_csv('history.csv')
    return model

def Train_re():
    
    load_model=tf.keras.models.load_model('Keras_Model_Netflix_epoch')
    
    # hist_re = load_model.fit([dataset.user_id, dataset.item_id], dataset.rating, epochs=1, verbose=1)
    hist_re = load_model.fit([dataset.User_ID, dataset.Movie_ID], dataset.User_Rating, steps_per_epoch=1000,epochs=5, verbose=1, workers=6)
    load_model.save('Keras_Model_Netflix_epoch_re',save_format='tf')

    loss_re=hist_re.history['loss']
    np_loss=np.array(loss_re)
    np.save('loss_re',np_loss)

    hist_re_df=pd.DataFrame(hist_re.history)
    print(hist_re_df)
    hist_re_df['epoch']=hist.epoch
    print(hist_re_df)
    hist_re_df.to_csv('history_re.csv')
    return load_model

def Get_Embedding_Matrix(model): 
    # 获得用户和项目的嵌入矩阵。
    user_embedding_learnt = model.get_layer(name='User-Embedding').get_weights()[0]
    movie_embedding_learnt = model.get_layer(name='Movie-Embedding').get_weights()[0]
    return user_embedding_learnt,movie_embedding_learnt

def User_Raw_ID_to_User_ID(user_raw_id):
    return User_IDs_Comparison_Table_dict[user_raw_id]

def Movie_Raw_ID_to_Movie_ID(movie_raw_id):
    return Movie_IDs_Comparison_Table_dict[movie_raw_id]

def User_ID_to_User_Raw_ID(user_id):
    return User_IDs_Comparison_Table['User_Raw_ID'][user_id]

def Movie_ID_to_Movie_Raw_ID(movie_id):
    return Movie_Titles_Comparison_Table['Movie_RAW_ID'][movie_id]

def Movie_Raw_ID_to_Movie_Name(movie_raw_id):
    return Movie_Raw_IDs_Names_Comparision_Comparison_Table_dict[movie_raw_id]

def Rating_Matrix(uel,mel):
    """
    mel=mel.T
    print(uel.shape)
    print(mel.shape)
    print("begin compute")
    Rating_Matrix_Predict=np.dot(uel,mel)
    for i in range(n_movies):
        Rating_Matrix_Predict.append((uel @ mel[i].T).tolist())
        print("Get {} movie".format(i))
    Rating_Matrix_Predict=np.array(Rating_Matrix_Predict)
    print(Rating_Matrix_Predict)
    """
    # np.save('Rating_Matrix_Predict.npy',Rating_Matrix_Predict)
    Rating_Matrix_Predict = np.load('Rating_Matrix_Predict.npy')
    # print(Rating_Matrix_Predict.shape)
    print('finish')
    return Rating_Matrix_Predict

def topN_recommend(user_raw_id, uel, mel, N):
    user_id = User_Raw_ID_to_User_ID(user_raw_id)
    movies_ratings = uel[user_id] @ mel.T # 导入tensorflow情况下，使用符号'@'进行矩阵乘法可以调动GPU，注意计算结果类型不是list。
    top_movies_id = np.argpartition(movies_ratings, -N)[-N:]
    # np.argpartition(原数组,N)返回一个列表，参数N指定了返回列表中索引为N的位置的数，为原数组中第N+1小的数的索引。而返回列表中排在索引N前面的索引对应的数都小于索引N对应的数，排在后面的索引对应的数都大于等于索引N对应的数，只划分数据，不关心顺序。
    # 因此此句代码仅得到了第1大到第N大的值对应原数组的索引列表（乱序），因此需要通过原数组的值大小重新进行排序。
    # 注意：np.sort()函数会使得每一行或列都进行排序，而我们需要获得的是"仅根据原数组的值大小"重新排序后的索引值的顺序。
    # 应当使用pandas的DataFrame来进行单列排序。
    # print(top_movies_id)
    top_movies_rating=[]
    for i in range(len(top_movies_id)):
        top_movies_rating.append([movies_ratings[top_movies_id[i]],top_movies_id[i]])
    # print(top_movies_rating)
    top_movies_id = list(pd.DataFrame(top_movies_rating,columns=['Ratings','Indexs']).sort_values('Ratings')['Indexs'])
    # print(top_movies_id)
    top_movies_rating = list(pd.DataFrame(top_movies_rating,columns=['Ratings','Indexs']).sort_values('Ratings')['Ratings'])
    # top_movies_rating
    top_Name = []
    for i in range(len(top_movies_id)):
        top_Name.append(("{}-th Movie ID:{} Predict Rating:{} Name:".format(i+1,str(top_movies_id[len(top_movies_id)-(i+1)]+1),str(top_movies_rating[len(top_movies_id)-(i+1)]))+Movie_Raw_ID_to_Movie_Name(top_movies_id[len(top_movies_id)-(i+1)]+1)))
    # 电影的索引值即为电影ID，电影ID+1即为电影RawID。
    return top_Name

def user_rating_for_movie(user_raw_id,movie_raw_id,uel,mel):
    user_id = User_Raw_ID_to_User_ID(user_raw_id)
    movie_id = Movie_Raw_ID_to_Movie_ID(movie_raw_id)
    return uel[user_id] @ mel[movie_id].T

def Euclidean_Distance(vec_1,vec_2):
    # ED = np.sqrt(np.sum((vec_1-vec_2)**2))
    return np.sum((np.array(vec_1)-np.array(vec_2))**2)

def similar_User_Raw_ID(user_raw_id,uel,mel, N):
    user_id = User_Raw_ID_to_User_ID(user_raw_id)
    N = N+1 # 不考虑自己
    n_users = 480189
    movies_ratings = uel[user_id].tolist()
    EDs=[]
    for i in range(n_users):
        EDs.append(Euclidean_Distance(movies_ratings,uel[i].tolist())) # 直接求用户矩阵（隐变量）的欧氏距离，而不是总的评分矩阵（含义是一样的，也是用户偏好）。
        # print("Get {} ed".format(i))
    # print(EDs)
    EDs=np.array(EDs)
    top_similar_user_id = np.argpartition(EDs,N)[0:N]
    # print(top_similar_user_id)
    top_similar_EDs=[]
    for i in range(len(top_similar_user_id)):
        top_similar_EDs.append([EDs[top_similar_user_id[i]],top_similar_user_id[i]])
    return [User_ID_to_User_Raw_ID(i) for i in list(pd.DataFrame(top_similar_EDs,columns=['EDs','Indexs']).sort_values('EDs')['Indexs'])[1:]]

def similar_Movie_Raw_ID(movie_raw_id,uel,mel, N):
    movie_id = Movie_Raw_ID_to_Movie_ID(movie_raw_id)
    N = N+1
    n_movies = 17770
    users_ratings = mel[movie_id].tolist()
    EDs=[]
    for i in range(n_movies):
        EDs.append(Euclidean_Distance(users_ratings,mel[i].tolist())) # 直接求物品矩阵（隐变量）的欧氏距离，而不是总的评分矩阵（含义是一样的，也是物品属性）。
        # print("Get {} ed".format(i))
    # print(EDs)
    EDs=np.array(EDs)
    top_similar_movie_id = np.argpartition(EDs,N)[0:N]
    # print(top_similar_movie_id)
    top_similar_EDs=[]
    for i in range(len(top_similar_movie_id)):
        top_similar_EDs.append([EDs[top_similar_movie_id[i]],top_similar_movie_id[i]])
    return [Movie_ID_to_Movie_Raw_ID(i) for i in list(pd.DataFrame(top_similar_EDs,columns=['EDs','Indexs']).sort_values('EDs')['Indexs'])[1:]]

Train()
Model=tf.keras.models.load_model('Keras_Model_Netflix_epoch')
uel,mel=Get_Embedding_Matrix(Model)

user_raw_id = 6 # 用户RawID最小为6。
movie_raw_id = 14961 # 用户RawID为6的第1推荐的电影的RawID，测试用。

"""
0.对某用户的前N个最推荐电影。
1.某用户对某电影的评分。
2.相似用户。
3.相似电影。
"""

print("-----Result-----")

# Rating_Matrix——Predict = Rating_Matrix(uel,mel)
# print(Rating_Matrix(uel,mel))

topN = topN_recommend(user_raw_id, uel, mel, N=10)
print("The top10 Recommand for user ID {}:".format(user_raw_id))
for i in topN:
    print(i)
print("-----------------")

user_rating_for_movie=user_rating_for_movie(user_raw_id, movie_raw_id, uel, mel)
print("The Rating from user ID {} to movie ID {}:".format(user_raw_id,movie_raw_id),user_rating_for_movie)
print("-----------------")

topN_similar_User_Raw_ID = similar_User_Raw_ID(user_raw_id, uel, mel, N=10)
print("The top10 Similar users for user ID {}".format(user_raw_id))
for i in topN_similar_User_Raw_ID:
    print(i)
print("-----------------")

topN_similar_Movie_Raw_ID = similar_Movie_Raw_ID(movie_raw_id, uel, mel, N=10)
topN_similar_Movie_Name = ["{}-th Movie ID:{} Name:".format(i+1,topN_similar_Movie_Raw_ID[i])+Movie_Raw_ID_to_Movie_Name(topN_similar_Movie_Raw_ID[i]) for i in range(len(topN_similar_Movie_Raw_ID))]
print("The top10 Similar movies for movies ID {}".format(movie_raw_id))
for i in topN_similar_Movie_Name:
    print(i)
print("-----------------")

def Plot_RSME(history):
    epoch = history['epoch']
    rmse = history['rmse']
    plt.plot(epoch,rmse)
    plt.xlabel("epochs")
    plt.ylabel("RMSE")
    plt.show()
    print("The Final RMSE:",list(rmse)[-1])

history = pd.read_csv('history.csv')

Plot_RSME(history)
print("-----------------")
