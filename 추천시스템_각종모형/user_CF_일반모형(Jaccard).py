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
from scipy.sparse import csr_matrix
from sklearn import preprocessing
import numpy as np

def topMatches(person,R):
	scores =[]
	neighbor_idx =np.argsort(R.tolist()[0])[::-1][:300]

	for user,score in zip(neighbor_idx,R[0,neighbor_idx].tolist()[0]):
		if score>0 and person !=user:
			scores.append((score,user))
	return scores

def pairwise_jaccard(X):
	X=X.astype(bool).astype(int)
	intrsct=X.dot(X.T)
	row_sums=intrsct.diagonal()
	unions=row_sums[:,None]+row_sums-intrsct
	dist=intrsct/unions
	return dist







random.seed(0)

dsn = cx_Oracle.makedsn("172.17.11.65",5701,"PORID022")
conn = cx_Oracle.connect("opr2001","prod_opr2001",dsn)
cur = conn.cursor()

#os.environ["OPENBLAS_NUM_THREADS"]="4"
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./recent_stock.log', level =logging.DEBUG)
start = time.time()

#kf = ShuffleSplit(n_splits=10,random_state=0)
datediff = (datetime.today() - timedelta(days=60)).strftime('%Y%m%d')


sql= "select a.trde_ymd , b.cust_no , a.stbd_code from aba.aba221tb00 a, aaa.aaa002mb00 b where a.trde_ymd >= "+datediff+    " and a.sell_buy_tp_code = '2' and a.cntr_qty > 0 and a.acct_no = b.acct_no(+) and a.secur_tp_code <> '2' and SUBSTR(a.acct_gds_code,1,1) IN ('0','2')"

data = pd.read_sql(sql,con=conn)
data.columns = ['date','user_id','item_id']
data=data.dropna()
data=data.drop_duplicates()

# condition : User must buy at least three shares
dates = [0]




active_user = [i[0] for i in data[['user_id','item_id']].drop_duplicates().groupby(data.user_id).count().itertuples() if i[1] >=5]
train_data = data.loc[data.date <='20190300']
#vmany_1month = data.loc[(data.date>='20190120')& (data.date<='20190200')]
test_data = data.loc[(data.date >='20190300')&(data.date<='20190400')]

le = preprocessing.LabelEncoder()
le.fit(data['item_id'])
le2 = preprocessing.LabelEncoder()
le2.fit(data['user_id'])


train_data=train_data[['user_id','item_id']].drop_duplicates()
test_data=test_data[['user_id','item_id']].drop_duplicates()



train_data['iid'] =le.transform(train_data['item_id'])
train_data['rating']=1
train_data['uid']=le2.transform(train_data['user_id'])

test_data['iid'] =le.transform(test_data['item_id'])
test_data['rating']=1
test_data['uid']=le2.transform(test_data['user_id'])



train_data=train_data[['uid','iid','rating']]
test_data=test_data[['uid','iid','rating']]

mtx = csr_matrix((train_data.rating,(train_data.uid,train_data.iid)))
sim_mtx = pairwise_jaccard(mtx)

common_user = list(set(train_data.uid.unique().tolist()).intersection(test_data.uid.unique().tolist()))

score_cnt5=0
total_cnt=0

for uid in common_user:
	
	# neighbors
	nbr = topMatches(uid,sim_mtx[uid])
	nbr = pd.DataFrame(nbr,columns=['score','uid'])
	
	#top score funds
	id_list=nbr.uid.tolist()
	join_table = train_data[train_data['uid'].isin(id_list)].join(nbr.set_index('uid'),on='uid')
	top_funds = join_table[['iid','score']].groupby(['iid']).agg('sum').sort_values(by=['score'],ascending=False)
	top_funds['iid']=top_funds.index
	top_funds.columns = ['score','iid']
	# duplicated
	duplicated = train_data[train_data.uid==uid].iid.unique().tolist()
	


	Reco_list5=[]
	total_cnt +=1

	for idx,data in enumerate(top_funds[~top_funds['iid'].isin(duplicated)].itertuples()):
		if idx<10:
			Reco_list5.append(data[2])
	if any(elem in Reco_list5 for elem in test_data[test_data.uid==uid].iid.unique().tolist()):
		score_cnt5 +=1.0


print(score_cnt5/total_cnt)
print(score_cnt5)
print(total_cnt)


"""

					logging.info("====================================================================")
					logging.info('total_cnt : ' + str(total_cnt))
					logging.info('factor : '+ str(factor) +' reg : '+ str(reg) + ' alpha : ' +str(alpha))
					logging.info('mean item_len ' + str(sum(length)/len(length)))
					logging.info('mean rank percentile ' + str(sum(mean_rank)/len(mean_rank)))
					logging.info('TOP-5 target rate ' + str(top5_ratio[0]))
					logging.info('TOP-5 precision ' + str(sum(precision)/len(precision)))
					logging.info('1week many buy fund target rate ' + str(score_many_cnt5/float(total_cnt)))
					logging.info("====================================================================")
"""
