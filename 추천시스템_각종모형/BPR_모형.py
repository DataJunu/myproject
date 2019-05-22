# -*- coding: utf8-*-
import warnings
import implicit
import pandas as pd
import scipy.sparse as sparse
from sklearn import preprocessing

warnings.filterwarnings("ignore")
from sklearn.model_selection import ShuffleSplit
import pymysql
import sys
from datetime import datetime, timedelta
import random

conn = pymysql.connect(host='localhost', user='root', password='', db='Stock_RS', charset='utf8', )
curs = conn.cursor(pymysql.cursors.DictCursor)

kf = ShuffleSplit(n_splits=5, random_state=1)

random.seed(10)


factors = [32 * i for i in range(2,20)]
regularizations = [0.0001]
iterations = [20]
alphas = [1]
dts = [30]
min_s = [4]
for dt in dts:

    temp = (datetime.strptime('20190130', '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
    """
    train_sql = "select user_id,item_id,count(item_id) as cnts from(\
    select distinct date,user_id,item_id from stock_tr\
     where (date > " + temp + " ) and user_id is not null) a\
     group by a.user_id,item_id\
    "
    """
    train_data = "select distinct user_id, item_id from stock_tr where date > " + temp
    data = pd.read_sql_query(sql=train_data, con=conn)
    """
    cnt_table = data.groupby(data.item_id).count()['user_id'].sort_values(ascending=False)
    item_len = (len(data.item_id.drop_duplicates().tolist()))
    frequent = cnt_table.index.tolist()[:(int(item_len * 0.01))]
    data = data.loc[~(data.item_id.isin(frequent))]
    """

    active_user = [i[0] for i in data[['user_id', 'item_id']].groupby(data.user_id).count().itertuples() if i[1] >= 4]

    data = data.loc[(data.user_id.isin(active_user))]

    le = preprocessing.LabelEncoder()
    le2 = preprocessing.LabelEncoder()

    le.fit(data['item_id'])
    le2.fit(data['user_id'])

    data['iid'] = le.transform(data['item_id'])
    data['uid'] = le2.transform(data['user_id'])
    data['cnts']=1
    data = data[['uid', 'iid', 'cnts']]

    for iter in iterations:
        for regularization in regularizations:
            for alpha in alphas:
                for factor in factors:

                    mean_rank = []
                    top5_ratio = []
                    top10_ratio = []
                    mean_len = []
                    precision = []
                    recall=[]
                    for train_index, validation_index in kf.split(data):

                        train_data = data.loc[data.index.isin(train_index)]
                        validation_data = data.loc[data.index.isin(validation_index)]

                        item_len = (len(train_data.iid.drop_duplicates().tolist()))

                        # item id는 전체를 기준으로 해줘야 한다.

                        sparse_item_user = sparse.csr_matrix(
                            (train_data['cnts'].astype(float), (train_data['iid'], train_data['uid'])))
                        sparse_user_item = sparse.csr_matrix(
                            (train_data['cnts'].astype(float), (train_data['uid'], train_data['iid'])))

                        temp_common_user = list(
                            set(train_data.uid.unique().tolist()).intersection(
                                validation_data.uid.unique().tolist()))

                        common_user = random.sample(temp_common_user, int(len(temp_common_user) / 10))

                        """
                        factors (int, optional) – The number of latent factors to compute
                        learning_rate (float, optional) – The learning rate to apply for SGD updates during training
                        regularization (float, optional) – The regularization factor to use
                        dtype (data-type, optional) – Specifies whether to generate 64 bit or 32 bit floating point factors
                        use_gpu (bool, optional) – Fit on the GPU if available
                        iterations (int, optional) – The number of training epochs to use when fitting the data

                        """

                        model = implicit.bpr.BayesianPersonalizedRanking(factors=factor, regularization=regularization,
                                                                     iterations=iter,learning_rate = 0.01)

                        # Calculate the confidence by multiplying it by our alpha value.
                        # 15 - 40
                        alpha_val = alpha
                        data_conf = (sparse_item_user * alpha_val).astype('double')

                        # Fit the model
                        model.fit(data_conf)
                        total_cnt = 0
                        score_cnt5 = 0
                        score_cnt10 = 0
                        temp_precision = []
                        temp_recall = []

                        """
                        정확도: 검색 결과로 가져온 문서 중 실제 관련된 문서 비율
                        재현율: 관련 문서 중 검색된 문서 비율
                        예
                        100개 문서를 가진 검색엔진에서
                        '범죄도시' 라는 키워드로 검색시, 검색 결과로 20개 문서가 나온 경우
                        이 때 20개 문서 중 16개 문서가 실제 '범죄도시' 와 관련된 문서이고
                        전체 100개 문서 중 '범죄도시' 와 관련된 총 문서는 32개라고 하면
                        정확도: 16 / 20 = 0.8

                        """

                        for uid in common_user:
                            try:
                                target_data = validation_data[validation_data.uid == uid]
                                target_item = target_data.iid.unique().tolist()
                                if not target_item:
                                    continue
                                else:

                                    r_ui_rank_prod = []
                                    r_ui_sum = []
                                    recommended = model.recommend(uid, sparse_user_item, N=item_len)
                                    total_cnt += 1
                                    item_set = list(set(target_item))
                                    # print(recommended[:10])
                                    ranked = [idx for idx, score in recommended]

                                    items = ranked[:10]
                                    # print(items)

                                    if any(elem in items for elem in target_item):
                                        score_cnt10 += 1.0
                                        common_item = list(
                                            set(items).intersection(
                                                target_item))
                                        temp_precision.append(len(common_item) / 10.0)
                                        temp_recall.append(len(common_item) / len(target_item))
                                    for iid in item_set:
                                        try:
                                            r_ui = target_data.loc[(target_data.iid == iid)].cnts.values[0]
                                            rank = (ranked.index(iid) + 1) / (item_len - len(target_item))
                                            r_ui_rank_prod.append(r_ui * rank)
                                            r_ui_sum.append(r_ui)
                                        except:
                                            continue
                                    if len(r_ui_sum) > 0:
                                        mean_rank.append(sum(r_ui_rank_prod) / sum(r_ui_sum))

                            except IndexError:
                                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
                                continue
                        # mean_len.append((len(active_user)))
                        precision.append(sum(temp_precision) / len(temp_precision))
                        recall.append(sum(temp_recall) / len(temp_recall))

                        top5_ratio.append(score_cnt10 / total_cnt)
                        # top10_ratio.append(score_cnt10 / total_cnt)
                    print(total_cnt)
                    print("factor : ", factor, ", regulization :", regularization, ", alpha :", alpha)
                    print("TOP - 10 ratio", sum(top5_ratio) / len(top5_ratio))
                    print("mean precision ", sum(precision) / len(precision))
                    print("mean rank ", sum(mean_rank) / len(mean_rank))
                    print("mean recall ", sum(recall) / len(recall))
