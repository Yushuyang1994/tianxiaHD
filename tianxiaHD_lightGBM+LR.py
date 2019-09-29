# -*- coding: utf-8 -*-

import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from multiprocessing import cpu_count
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 将lighgbm评价函数置为f1_score
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


# 选择最优参数，使用时需要将main函数中的注释去掉
def find_params(train_x, train_y):
    best_params = {}
    
    # cross validation, 随机划分成7:3
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, random_state=1, test_size=0.3)

    # 设置初始参数
    print('set initial parameters')
    params = {
        'boosting_type' : 'gbdt',
        'objective' : 'binary',
        'learning_rate' : 0.1,
        'num_leaves' : 20,
        'max_depth' : 7,
        'subsample' : 0.7,
        'subsample_freq' : 1,
        'colsample_bytree' : 0.8,
        'max_bin': 255,
        'min_data_in_leaf': 61,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.9,
        'bagging_freq': 25,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'min_split_gain': 0.7,
        'n_estimators': 63,
        'random_state' : 2019,
        'n_jobs' : cpu_count() - 1
    }

    # 调整n_estimators
    data_train = lgb.Dataset(X_train, y_train)
    cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=True, shuffle=True,
                        feval=lgb_f1_score, early_stopping_rounds=50, seed=1)
    # print(cv_results)
    print('best n_estimators:', len(cv_results['f1-mean']))
    print('best cv score:', pd.Series(cv_results['f1-mean']).max())
    params['n_estimators'] = len(cv_results['f1-mean'])
    best_params['n_estimators'] = len(cv_results['f1-mean'])
    max_f1 = pd.Series(cv_results['f1-mean']).max()

    # 调整num_leaves和max_depth
    print("set num_leaves and max_depth")
    for num_leaves in range(5, 100, 5):
        for max_depth in range(3, 8, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth

            cv_results = lgb.cv(
                params,
                data_train,
                seed=1,
                nfold=5,
                feval=lgb_f1_score,
                early_stopping_rounds=10,
                verbose_eval=True
            )

            mean_f1_score = pd.Series(cv_results['f1-mean']).max()
            boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()
            if mean_f1_score >= max_f1:
                max_f1 = mean_f1_score
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']

    # 调整max_bin和min_data_in_leaf
    print("set max_bin and min_data_in_leaf")
    for max_bin in range(5, 506, 10):
        for min_data_in_leaf in range(1, 102, 10):
            params['max_bin'] = max_bin
            params['min_data_in_leaf'] = min_data_in_leaf

            cv_results = lgb.cv(
                params,
                data_train,
                seed=1,
                nfold=5,
                feval=lgb_f1_score,
                early_stopping_rounds=10,
                verbose_eval=True
            )

            mean_f1_score = pd.Series(cv_results['f1-mean']).max()
            boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()

            if mean_f1_score >= max_f1:
                max_f1 = mean_f1_score
                best_params['max_bin'] = max_bin
                best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']

    # 调整feature_fraction,bagging_fraction和bagging_freq
    print("set feature_fraction, bagging_fraction and bagging_freq")
    for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for bagging_freq in range(0, 50, 5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq

                cv_results = lgb.cv(
                    params,
                    data_train,
                    seed=1,
                    nfold=5,
                    feval=lgb_f1_score,
                    early_stopping_rounds=10,
                    verbose_eval=True
                )

                mean_f1_score = pd.Series(cv_results['f1-mean']).max()
                boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()

                if mean_f1_score >= max_f1:
                    max_f1 = mean_f1_score
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq

    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']

    # 调整lambda_l1和lambda_l2
    print("set lambda_l1 and lambda_l2")
    for lambda_l1 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for lambda_l2 in [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            cv_results = lgb.cv(
                params,
                data_train,
                seed=1,
                nfold=5,
                feval=lgb_f1_score,
                early_stopping_rounds=10,
                verbose_eval=True
            )

            mean_f1_score = pd.Series(cv_results['f1-mean']).max()
            boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()

            if mean_f1_score >= max_f1:
                max_f1 = mean_f1_score
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']

    # 调整min_split_gain
    print("set min_split_gain")
    for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        params['min_split_gain'] = min_split_gain

        cv_results = lgb.cv(
            params,
            data_train,
            seed=1,
            nfold=5,
            feval=lgb_f1_score,
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_f1_score = pd.Series(cv_results['f1-mean']).max()
        boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()

        if mean_f1_score >= max_f1:
            max_f1 = mean_f1_score
            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params['min_split_gain'] = best_params['min_split_gain']

    print(best_params)


def main():
    paths = ['./train_data/000000_0', './test_data/000000_0']
    headers = ['''role_id	device_model	os_name	os_ver	app_channel	app_ver	server	login_times_week	
    time_after_create	day1item500000	day1item500001	day1item500002	day1item510000	day1item510001	
    day1item510002	day1item510003	day1item510007	day1item520091	day1item525002	day1item581000	day1item581001	
    day1item583001	day1item583002	day1item583003	day1item584003	day1item590603	day2item500000	day2item500001	
    day2item500002	day2item510000	day2item510001	day2item510002	day2item510003	day2item510007	day2item520091	
    day2item525002	day2item581000	day2item581001	day2item583001	day2item583002	day2item583003	day2item584003	
    day2item590603	day3item500000	day3item500001	day3item500002	day3item510000	day3item510001	day3item510002	
    day3item510003	day3item510007	day3item520091	day3item525002	day3item581000	day3item581001	day3item583001	
    day3item583002	day3item583003	day3item584003	day3item590603	day4item500000	day4item500001	day4item500002	
    day4item510000	day4item510001	day4item510002	day4item510003	day4item510007	day4item520091	day4item525002	
    day4item581000	day4item581001	day4item583001	day4item583002	day4item583003	day4item584003	day4item590603	
    day5item500000	day5item500001	day5item500002	day5item510000	day5item510001	day5item510002	day5item510003	
    day5item510007	day5item520091	day5item525002	day5item581000	day5item581001	day5item583001	day5item583002	
    day5item583003	day5item584003	day5item590603	day6item500000	day6item500001	day6item500002	day6item510000	
    day6item510001	day6item510002	day6item510003	day6item510007	day6item520091	day6item525002	day6item581000	
    day6item581001	day6item583001	day6item583002	day6item583003	day6item584003	day6item590603	day7item500000	
    day7item500001	day7item500002	day7item510000	day7item510001	day7item510002	day7item510003	day7item510007	
    day7item520091	day7item525002	day7item581000	day7item581001	day7item583001	day7item583002	day7item583003	
    day7item584003	day7item590603	day1item500000free	day1item500001free	day1item500002free	day1item510000free	
    day1item510001free	day1item510002free	day1item510003free	day1item510007free	day1item520091free	
    day1item525002free	day1item581000free	day1item581001free	day1item583001free	day1item583002free	
    day1item583003free	day1item584003free	day1item590603free	day2item500000free	day2item500001free	
    day2item500002free	day2item510000free	day2item510001free	day2item510002free	day2item510003free	
    day2item510007free	day2item520091free	day2item525002free	day2item581000free	day2item581001free	
    day2item583001free	day2item583002free	day2item583003free	day2item584003free	day2item590603free	
    day3item500000free	day3item500001free	day3item500002free	day3item510000free	day3item510001free	
    day3item510002free	day3item510003free	day3item510007free	day3item520091free	day3item525002free	
    day3item581000free	day3item581001free	day3item583001free	day3item583002free	day3item583003free	
    day3item584003free	day3item590603free	day4item500000free	day4item500001free	day4item500002free	
    day4item510000free	day4item510001free	day4item510002free	day4item510003free	day4item510007free	
    day4item520091free	day4item525002free	day4item581000free	day4item581001free	day4item583001free	
    day4item583002free	day4item583003free	day4item584003free	day4item590603free	day5item500000free	
    day5item500001free	day5item500002free	day5item510000free	day5item510001free	day5item510002free	
    day5item510003free	day5item510007free	day5item520091free	day5item525002free	day5item581000free	
    day5item581001free	day5item583001free	day5item583002free	day5item583003free	day5item584003free	
    day5item590603free	day6item500000free	day6item500001free	day6item500002free	day6item510000free	
    day6item510001free	day6item510002free	day6item510003free	day6item510007free	day6item520091free	
    day6item525002free	day6item581000free	day6item581001free	day6item583001free	day6item583002free	
    day6item583003free	day6item584003free	day6item590603free	day7item500000free	day7item500001free	
    day7item500002free	day7item510000free	day7item510001free	day7item510002free	day7item510003free	
    day7item510007free	day7item520091free	day7item525002free	day7item581000free	day7item581001free	
    day7item583001free	day7item583002free	day7item583003free	day7item584003free	day7item590603free	day1cash	
    day2cash	day3cash	day4cash	day5cash	day6cash day7cash	day1opjiahu	day1opkaikong	day1opxiangqian	
    day1opduanzao	day2opjiahu	day2opkaikong	day2opxiangqian	day2opduanzao	day3opjiahu	day3opkaikong	
    day3opxiangqian	day3opduanzao	day4opjiahu	day4opkaikong	day4opxiangqian	day4opduanzao	day5opjiahu	
    day5opkaikong	day5opxiangqian day5opduanzao	day6opjiahu	day6opkaikong	day6opxiangqian	day6opduanzao	
    day7opjiahu	day7opkaikong	day7opxiangqian	day7opduanzao	day1itxiangqian day2itxiangqian	day3itxiangqian	
    day4itxiangqian	day5itxiangqian	day6itxiangqian day7itxiangqian	day1ronglian	day2ronglian	day3ronglian	
    day4ronglian	day5ronglian	day6ronglian	day7ronglian	day1ronglianbaoshi	day2ronglianbaoshi	
    day3ronglianbaoshi	day4ronglianbaoshi	day5ronglianbaoshi	day6ronglianbaoshi	day7ronglianbaoshi	
    day1task	day2task day3task	day4task	day5task	day6task	day7task	day1battlewin	day2battlewin	
    day3battlewin	day4battlewin	day5battlewin	day6battlewin	day7battlewin	day1battlelose	day2battlelose	
    day3battlelose	day4battlelose	day5battlelose	day6battlelose	day7battlelose	day1combatwin	day2combatwin	
    day3combatwin	day4combatwin	day5combatwin	day6combatwin	day7combatwin	day1combatlose	day2combatlose	
    day3combatlose	day4combatlose	day5combatlose	day6combatlose	day7combatlose	day1end_state_1	day1end_state1	
    day1end_state2	day1end_state3	day1end_state4	day1end_state5	day1end_state6	day1end_state7	day2end_state_1	
    day2end_state1	day2end_state2	day2end_state3	day2end_state4	day2end_state5	day2end_state6	day2end_state7	
    day3end_state_1 day3end_state1	day3end_state2	day3end_state3	day3end_state4	day3end_state5	day3end_state6	
    day3end_state7	day4end_state_1	day4end_state1	day4end_state2	day4end_state3	day4end_state4	day4end_state5	
    day4end_state6	day4end_state7	day5end_state_1	day5end_state1	day5end_state2	day5end_state3	day5end_state4	
    day5end_state5	day5end_state6	day5end_state7	day6end_state_1	day6end_state1	day6end_state2	day6end_state3	
    day6end_state4	day6end_state5	day6end_state6	day6end_state7	day7end_state_1	day7end_state1	day7end_state2	
    day7end_state3	day7end_state4	day7end_state5	day7end_state6	day7end_state7	career label''']
    if os.path.exists('./train.csv') and os.path.exists('./test.csv'):
        pass
    elif os.path.exists('./train_data.csv') and os.path.exists('./test_data.csv'):
        train_data = pd.read_csv('./train_data.csv')
        test_data = pd.read_csv('./test_data.csv')
    else:
        for i in range(len(paths)):
            role_data = []
            column_names = headers[0].strip().split()
            with open(paths[i], 'r') as f:
                for j,line in enumerate(f):
                    line = line.strip('\n').split('\t')
                    line = [' '.join(sorted(s.split(',')))  for s in line]
                    role_dict = dict(zip(column_names, line))
                    role_data.append(role_dict)
                role_df = pd.DataFrame(role_data)
                role_df = role_df.replace('', np.nan)
                print("Write data to " + '/'.join(paths[i].strip().split('/')[:-1]) + ".csv")
                role_df.to_csv('/'.join(paths[i].strip().split('/')[:-1]) + ".csv", index = False)
        train_data = pd.read_csv('./train_data.csv')
        test_data = pd.read_csv('./test_data.csv')
    if os.path.exists('./train.csv') and os.path.exists('./test.csv'):
        train = pd.read_csv('./train.csv')
        test = pd.read_csv('./test.csv')
    else:
        train = train_data.copy()
        test = test_data.copy()
        paths_ = ['./left_yuanbaoall_1/000000_0', './left_yuanbaofree_1/000000_0', './left_yuanbaoall_2/000000_0',
             './left_yuanbaofree_2/000000_0', './role_level_day_1/000000_0', './role_level_day_2/000000_0',
             './role_force_day_1/000000_0', './role_force_day_2/000000_0', './role_friends_day_1/000000_0',
             './role_friends_day_2/000000_0', './role_wing_day_1/000000_0', './role_wing_day_2/000000_0']
        for path_ in paths_:
            if '_1' in path_:
                tmp = pd.read_csv(path_.replace('/000000_0', '.csv'))
                train = pd.merge(train, tmp, on = "role_id", how = "left")
            elif '_2' in path_:
                tmp = pd.read_csv(path_.replace('/000000_0', '.csv'))
                test = pd.merge(test, tmp, on = 'role_id', how = 'left')
        train.to_csv("./train.csv", index = False)
        test.to_csv("./test.csv", index = False)
    print('raw_data ready! prepare to one_hot and cv.')
    one_hot_features = []
    vector_features = ['device_model', 'os_name', 'os_ver', 'app_channel', 'app_ver', 'server', 'day1itxiangqian',
                       'day2itxiangqian', 'day3itxiangqian', 'day4itxiangqian', 'day5itxiangqian', 'day6itxiangqian',
                       'day7itxiangqian']
    other_features = [x for x in headers[0].strip().split()[1:-1] if x not in (vector_features + one_hot_features)]
    test['label'] = -1
    data = pd.concat([train, test])
    train_x = train[other_features].fillna(0)
    test_x = test[other_features].fillna(0)
    train_y = train.pop('label')
    # print(train_x.shape)
    # one_hot process
    enc = OneHotEncoder()
    for feature in one_hot_features:
        enc.fit(data[feature].values.reshape(-1, 1))
        train_feature = enc.transform(train[feature].values.reshape(-1, 1))
        test_feature = enc.transform(test[feature].values.reshape(-1, 1))
        train_x = sparse.hstack((train_x, train_feature))
        test_x = sparse.hstack((test_x, test_feature))
    print('one-hot ready!')

    # cv process, 增加这一项后f1score有少许提升
    cv = CountVectorizer()
    for feature in vector_features:
        try:
            cv.fit(data[feature].values.astype('U'))
        except:
            print(feature)
            raise ValueError('error')
        train_feature = cv.transform(train[feature].values.astype('U'))
        train_x = sparse.hstack((train_x, train_feature))
        test_feature = cv.transform(test[feature].values.astype('U'))
        test_x = sparse.hstack((test_x, test_feature))
    # print(train_x.toarray().shape)
    res = pd.DataFrame(test['role_id'])

    print('CountVectorizer ready!')
    # find_params(train_x, train_y)

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, random_state=1, test_size=0.3)
    data_train = lgb.Dataset(X_train, y_train)
    params = {
        'boosting_type' : 'gbdt',
        'objective' : 'binary',
        'learning_rate' : 0.01,
        'num_leaves' : 22,
        'max_depth' : 7,
        'max_bin': 255,
        'min_data_in_leaf': 61,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.9,
        'bagging_freq': 25,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'min_split_gain': 0.7,
        'random_state' : 2019,
        'n_estimators' : 984,
        'n_jobs' : cpu_count() - 1
    }
    # 模型最终参数
    model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', learning_rate=0.01, n_estimators=1000,
                               num_leaves=22, max_bin=255, min_data_in_leaf=61, bagging_fraction=0.9, bagging_freq=25,
                               feature_fraction= 0.7)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    y_pre_prob1 = model.predict_proba(X_test)[:,1]
    # 划分的测试集上的分数
    print("precision_score: ", precision_score(y_test, y_pre))# precision_score: 0.748
    print("recall_score: ", recall_score(y_test, y_pre))# recall_score: 0.580
    print("f1_score: ", f1_score(y_test, y_pre))# f1_score: 0.653
    # 优化lr的正则项
    # max_f1 = 0
    # best_c = 0
    # for c in range(1,1000,10):
    #     lr = LogisticRegression(C=c, random_state=0, class_weight='balanced', n_jobs=-1)
    #     lr.fit(X_train, y_train)
    #     y_pre2 = lr.predict(X_test)
    #     if f1_score(y_test, y_pre2) >= max_f1:
    #         max_f1 = f1_score(y_test, y_pre2)
    #         best_c = c
    #     print("precision_score: ", precision_score(y_test, y_pre2))# precision_score:
    #     print("recall_score: ", recall_score(y_test, y_pre2))# recall_score:
    #     print("f1_score: ", f1_score(y_test, y_pre2))# f1_score:
    # print(best_c)
    lr = LogisticRegression(C=841, random_state=0, class_weight='balanced', n_jobs=-1)
    lr.fit(X_train, y_train)
    y_pre2 = lr.predict(X_test)
    y_pre_prob2 = lr.predict_proba(X_test)[:, 1]
    print("precision_score: ", precision_score(y_test, y_pre2))# precision_score: 0.446
    print("recall_score: ", recall_score(y_test, y_pre2))# recall_score: 0.795
    print("f1_score: ", f1_score(y_test, y_pre2))# f1_score: 0.572

    #使用lightgbm预测的结果召回率较低，而lr预测的结果则准确率较低，因此考虑将两者预测结果结合从而达到更好的f1_score
    y_pre3 = np.rint(2 * y_pre_prob1 * y_pre_prob2 / (y_pre_prob1 + y_pre_prob2))
    # y_pre3 = np.rint((y_pre_prob1 + y_pre_prob2) / 2)
    print("precision_score: ", precision_score(y_test, y_pre3))# precision_score: 0.716
    print("recall_score: ", recall_score(y_test, y_pre3))# recall_score: 0.631
    print("f1_score: ", f1_score(y_test, y_pre3))# f1_score: 0.671 取平均值四舍五入0.667

    model.fit(train_x, train_y)
    label_prob1 = model.predict_proba(test_x)[:,1]
    lr.fit(train_x, train_y)
    label_prob2 = lr.predict_proba(test_x)[:, 1]
    res['label'] = np.rint(2 * label_prob1 * label_prob2 / (label_prob1 + label_prob2))
    # res['label'] = np.rint((label_prob1 + label_prob2) / 2)
    res = res[res.label != 0]
    res = res.drop('label', axis=1)
    res.to_csv('./result.txt', sep='\t', index=False, header=None)
    
    # 降低学习速率提高迭代次数来提高准确率
    # for learning_rate in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     params['learning_rate'] = learning_rate
    #     cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=True, shuffle=True,
    #                         feval=lgb_f1_score, early_stopping_rounds=50, seed=1)
    #     # print(cv_results)
    #     print('learning_rate:', learning_rate)
    #     print('best n_estimators:', len(cv_results['f1-mean']))
    #     print('best cv score:', pd.Series(cv_results['f1-mean']).max())
    # for learning_rate in [0.08, 0.09 ,0.1 ,0.11 ,0.12 ,0.13 ,0.14, 0.15]:
    #     params['learning_rate'] = learning_rate
    #     cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=True, shuffle=True,
    #                         feval=lgb_f1_score, early_stopping_rounds=50, seed=1)
    #     # print(cv_results)
    #     print('learning_rate:', learning_rate)
    #     print('best n_estimators:', len(cv_results['f1-mean']))
    #     print('best cv score:', pd.Series(cv_results['f1-mean']).max())

    # best_params = {'num_leaves': 22, 'max_depth': 7, 'max_bin': 255, 'min_data_in_leaf': 61, 'feature_fraction': 0.7,
    #                'bagging_fraction': 0.9, 'bagging_freq': 25, 'lambda_l1': 0.0, 'lambda_l2': 0.0,
    #                'min_split_gain': 0.7}
    # max_f1 = 0
    # cv_results = lgb.cv(
    #     params,
    #     data_train,
    #     seed=1,
    #     num_boost_round=2000,
    #     nfold=5,
    #     feval=lgb_f1_score,
    #     early_stopping_rounds=100,
    #     verbose_eval=True
    # )
    #
    # mean_f1_score = pd.Series(cv_results['f1-mean']).max()
    # boost_rounds = pd.Series(cv_results['f1-mean']).idxmax()
    # if mean_f1_score >= max_f1:
    #     max_f1 = mean_f1_score
    #     best_params['n_estimators'] = boost_rounds
    # print(max_f1, best_params)

if __name__ == '__main__':
    main()
