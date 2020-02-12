
# coding: utf-8

# In[1]:


from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql import SQLContext
import pandas as pd

#from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry


spark = SparkSession.builder.appName("hw1_011504272(method1)").getOrCreate()
data_rdd = spark.read.text("C:\\Users\\kpgaj\\train.dat").rdd
splitted = data_rdd.map(lambda row: row.value.split("\t"))
splitted.collect()
rrdd = splitted.map(lambda a: Row(UserID=int(a[0]),MovieID=int(a[1]), Rating=float(a[2]), Timestamp=int(a[3])))
data_frame_spark = spark.createDataFrame(rrdd).cache()
pandas_df = data_frame_spark.toPandas()


# In[2]:


pandas_df.head()


# In[3]:


u_matrix = (pandas_df.pivot(index="UserID",columns="MovieID",values="Rating").fillna(0))


# In[4]:


import time
import surprise


svd = surprise.SVD(random_state=2, n_factors=200, n_epochs=1000, verbose=True)
df_train= pandas_df.drop(columns = 'Timestamp')


# In[5]:


from surprise import Reader
reader = Reader()
df_set_train =surprise.Dataset.load_from_df(df_train[['MovieID', 'UserID', 'Rating']], reader) 


# In[6]:


from surprise.model_selection import cross_validate

cross_validate(svd, df_set_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[7]:


trainset = df_set_train.build_full_trainset()

svd.fit(trainset)


# In[8]:


list_actual =[]
list_pred =[]
for i in range(len(pandas_df)) :
    i = pandas_df.loc[i,"UserID"]
    j = pandas_df.loc[i,"MovieID"]
    k = pandas_df.loc[i,"Rating"]
    list_actual.append(k)
    list_pred.append(svd.predict(i,j,k).est)
    print(i,j,k, svd.predict(i,j,k)) 
    


# In[9]:


from sklearn.metrics import mean_squared_error
mean_squared_error(list_actual, list_pred)  


# In[10]:


data_test = spark.read.text("C:\\Users\\kpgaj\\test.dat")
splitted_t = data_rdd.map(lambda row: row.value.split("\t"))

rrdd_t = splitted_t.map(lambda a: Row(UserID=int(a[0]),MovieID=int(a[1])))
data_frame_spark_T = spark.createDataFrame(rrdd_t).cache()
pd_test = data_frame_spark_T.toPandas()


# In[11]:


pd_test.shape


# In[12]:


pd_test_sample = pd_test[:2154]


# In[13]:


pd_test_sample.shape


# In[14]:


f = open("format.dat","w+")
f.write("RatingID,Rating\n")
count=0
for i,row in pd_test_sample.iterrows():
    count += 1
    
    f.write('{},{:.2f}\n'.format(count,(svd.predict(row["MovieID"],row["UserID"]).est)))


f.close()

