import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cx_Oracle
import pandas as pd
import implicit
import scipy.sparse as sparse
from datetime import datetime,timedelta
import time
import socket
import getpass
import json
import sys
import logging
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
import cx_Oracle
import random
random.seed(0)
import pymysql
import sys
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, concatenate, multiply
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras import initializers
from sklearn.model_selection import ShuffleSplit


random.seed(1)

def init_normal(shape):
    return initializers.normal(shape, initializers.VarianceScaling(scale=0.01))


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer='random_normal', W_regularizer=l2(reg_mf),input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer='random_normal', W_regularizer=l2(reg_mf),input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_initializer='random_normal', W_regularizer=l2(reg_layers[0]),input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_initializer='random_normal', W_regularizer=l2(reg_layers[0]),input_length=1)


    # MF part
    #mf_user_latent = Reshape(target_shape=(mf_dim,))(MF_Embedding_User(user_input))
    #mf_item_latent = Reshape(target_shape=(mf_dim,))(MF_Embedding_Item(item_input))
    
    mf_user_latent=Flatten()(MF_Embedding_User(user_input))
    mf_item_latent=Flatten()(MF_Embedding_Item(item_input))





    mf_vector = multiply([mf_user_latent, mf_item_latent])  # element-wise multiply

    # MLP part
    #mlp_user_latent = Reshape(target_shape=(int(layers[0]/2),))(MLP_Embedding_User(user_input))
    #mlp_item_latent = Reshape(target_shape=(int(layers[0]/2),))(MLP_Embedding_Item(item_input))
    

    mlp_user_latent = Flatten()(MF_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MF_Embedding_Item(item_input))

    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name="prediction")(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model


def get_train_instances(train, num_negatives):
    user_input = list(train.nonzero()[0])
    item_input = list(train.nonzero()[1])
    labels = [1 for i in range(len(user_input))]
    negative = random.sample(np.argwhere(train==0).tolist(),k=int(train.shape[0])*num_negatives)

    for i in negative:
        user_input.append(i[0])
        item_input.append(i[1])
        labels.append(0)
    return user_input, item_input, labels


def get_random_item_list(num_negatives, num_items):
    random_list = [np.random.randint(num_items) for i in range(int(num_negatives))]
    
    #for t in range(num_negatives):
    #    j = np.random.randint(num_items)
    #    while j in item_list:
    #        j = np.random.randint(num_items)
    #    random_list.append(j)

    return random_list





dsn = cx_Oracle.makedsn("172.17.11.65",5701,"PORID022")
conn = cx_Oracle.connect("opr2001","prod_opr2001",dsn)
cur = conn.cursor()

#os.environ["OPENBLAS_NUM_THREADS"]="4"
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./Neural_CF.log', level =logging.DEBUG)
start = time.time()

kf = ShuffleSplit(n_splits=5,random_state=0)
datediff = (datetime.today() - timedelta(days=30)).strftime('%Y%m%d')


sql ="select distinct b.cust_no , a.stbd_code from aba.aba221tb00 a, aaa.aaa002mb00 b where a.trde_ymd >= "+datediff+" and a.sell_buy_tp_code = '2' and a.cntr_qty > 0 and a.acct_no = b.acct_no(+) and a.secur_tp_code <> '2' and SUBSTR(a.acct_gds_code,1,1) IN ('0','2')"
data= pd.read_sql(sql,con=conn)

data.columns=['user_id','item_id']
active_user = [i[0] for i in data[['user_id', 'item_id']].groupby(data.user_id).count().itertuples() if i[1] >= 4]

data = data.loc[(data.user_id.isin(active_user))]

le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

le.fit(data['item_id'])
le2.fit(data['user_id'])

data['iid'] = le.transform(data['item_id'])
data['uid'] = le2.transform(data['user_id'])
data['cnts'] = 1.0
data = data[['uid', 'iid','cnts']]
num_epochs = 20
batch_size = 256

mf_dim = [8*i for i in range(1,10)]
layers = eval(str([64, 32, 16, 8]))
reg_mf = [0]
reg_layers = eval(str([0,0,0,0]))
num_negatives = 4
learning_rate = 0.001
learner = 'sgd'
num_users = data.uid.max() + 1
num_items = data.iid.max() + 1
topk = 10
#print(data.head())
for dim in mf_dim:
    for reg in reg_mf:
        logging.info("====================================================================")
        logging.info("mf dim   "+str(dim))
        logging.info("reg mf   "+str(reg))
        import heapq
        
        top10_ratio = []
        mean_len = []
        precision = []
        recall = []
        import scipy.sparse as sparse
        
        for train_index, validation_index in kf.split(data):
            #print('fold')
            train_data = data.loc[data.index.isin(train_index)]
            validation_data = data.loc[data.index.isin(validation_index)]
            model = get_model(num_users, num_items, dim, layers, reg_layers, reg)

            if learner.lower() == "adagrad":
                model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "rmsprop":
                model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
            elif learner.lower() == "adam":
                model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
            else:
                model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
            #print(train_data.head())
            #print(validation_data.head())
            sparse_user_item = sparse.csr_matrix( (train_data['cnts'].astype(float), (train_data['uid'],train_data['iid'])))
                
            user_input, item_input, labels = get_train_instances(sparse_user_item, num_negatives)
            hist = model.fit([user_input, item_input], labels, epochs=20, verbose=2)
        
            temp_common_user = list(
                set(train_data.uid.unique().tolist()).intersection(
                    validation_data.uid.unique().tolist()))
        
            common_user = random.sample(temp_common_user, int(len(temp_common_user) / 10))
        
            total_cnt = 0
            score_cnt10 = 0
            temp_precision = []
            temp_recall = []
        
            for uid in common_user:
                try:
                    #print(common_user)
                    target_data = validation_data[validation_data.uid == uid]
                    target_item = target_data.iid.unique().tolist()
                    if not target_item:
                        continue
                    else:
                        # def get_random_item_list(item_list,num_negatives,num_items):
                        items =[i for i in range(int(num_items-1))]
                        #get_random_item_list(2000, num_items)
                        for a in target_item:
                            items.append(a)
                        #items = list(set(items))
                        items = list(set(items) - set(train_data[train_data.uid==uid].iid.unique().tolist()))
                        #print(items)
                        # Get prediction scores
                        map_item_score = {}
                        #users = np.full(len(items), int(uid), dtype='int32')
                        uids = [uid for i in range(len(items))]
        
        
        
                        predictions = model.predict([uids, items])
                        for i in range(len(items)):
                            item = items[i]
                            map_item_score[item] = predictions[i]
                        items.pop()
        
                        ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
                        #print(ranklist)
                        #print(target_item)
                        total_cnt += 1
                        if any(elem in ranklist for elem in target_item):
                            #print('yes')
                            score_cnt10 += 1.0
                            common_item = list(set(ranklist).intersection(target_item))
                            #print(common_item)
                            temp_precision.append(len(common_item) / topk)
                            #if len(target_item)>0:
                            temp_recall.append(len(common_item) / len(target_item))
        
        
                except:
                    print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))

            print("TOP-10 target rate : ", str(score_cnt10 / total_cnt))
            try:
                logging.info('TOP-10 target rate ' + str(score_cnt10/total_cnt))
                precision.append(sum(temp_precision) / len(temp_precision))
                recall.append(sum(temp_recall) / len(temp_recall))
                top10_ratio.append(score_cnt10 / total_cnt)
            except:
                continue
        try:
            print("TOP - 10 ratio ", sum(top10_ratio) / len(top10_ratio))
            print("mean precision  ", sum(precision) / len(precision))
            print("mean recall ", sum(recall) / len(recall))
            logging.info('total_cnt : ' + str(total_cnt))
            logging.info("mean TOP - 10 ratio "+ str(sum(top10_ratio) / len(top10_ratio)))
            logging.info("mean precision "+ str(sum(precision) / len(precision)))
            logging.info("mean recall "+ str(sum(recall) / len(recall)))
        except:
            continue
