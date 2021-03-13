import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,matthews_corrcoef,confusion_matrix
import sys,os
import pickle

#sys.path.append('../')
#from getDataStatistic import getPDBStat, getMutationStats, plot_pdbStat

def getScores(y_label, y_predictions):
    '''
    This function takes model and data and returns scores
    '''
    accuracy = accuracy_score(y_label, y_predictions)
    precision = precision_score(y_label, y_predictions)
    f1 = f1_score(y_label, y_predictions)
    recall = recall_score(y_label, y_predictions)
    mcc = matthews_corrcoef(y_label, y_predictions)

    print '%10s : %8.3f' % ('accuracy', accuracy)
    print '%10s : %8.3f' % ('precision', precision)
    print '%10s : %8.3f' % ('recall', recall)
    print '%10s : %8.3f' % ('f1', f1)
    print '%10s : %8.3f' % ('mcc', mcc)

    if len(confusion_matrix(y_label, y_predictions).ravel())<4:
        print "WARNING!!! BUG IN CONFUSION MATRIX"
        return [accuracy,precision,recall,f1,mcc]

    tn, fp, fn, tp = confusion_matrix(y_label, y_predictions).ravel()
    s = (tn+fp+fn+tp)/100.0
    print "tn, fp, fn, tp:", (tn, fp, fn, tp)
    print "tn, fp, fn, tp:", (tn/s, fp/s, fn/s, tp/s)

    return [accuracy,precision,recall,f1,mcc]

def processData(data, feature_set = "all"):

    '''
    Here I pre-process the input .csv
    :param feature_set: set of features to keep. Options: all, sequence,structure,energy,reduced
    :return:
    '''
    mutations_data = data

    if 'redundancy' in mutations_data:
        mutations_data = mutations_data.drop('redundancy', axis=1)

    if 'class' in mutations_data:
        mutations_data = mutations_data.rename(columns={'class': 'Diseased'})
        mutations_data.loc[mutations_data['Diseased'] == 2, 'Benign'] = 1
        mutations_data.loc[mutations_data['Diseased'] == 1, 'Benign'] = 0
        mutations_data.loc[mutations_data['Diseased'] == 2, 'Diseased'] = 0

    mutations_data_without_class = mutations_data

    mutations_data_without_class = mutations_data_without_class.drop('Diseased', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('Benign', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('separator', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('name', axis=1)

    # DROP SOME FEATURES
    mutations_data_without_class = mutations_data_without_class.drop('nHelix', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('seq_length', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('nStrand', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('nTM', axis=1)
    mutations_data_without_class = mutations_data_without_class.drop('nTurn', axis=1)


    all_features = mutations_data_without_class.columns.values.tolist()

    sequence_features = ['wt_hydrophobicity_scale','wt_amphiphilicity_index','wt_bulkiness','wt_polarity',
                         'wt_polarizability_parameter','wt_isoelectric_point','wt_asamax','wt_n_hb','wt_charge',
                         'wt_radius_gyration','wt_composition_membrane','wt_contribution_stability',
                         'mut_hydrophobicity_scale','mut_amphiphilicity_index','mut_bulkiness','mut_polarity',
                         'mut_polarizability_parameter','mut_isoelectric_point','mut_asamax','mut_n_hb',
                         'mut_charge','mut_radius_gyration','mut_composition_membrane','mut_contribution_stability',
                         'blosum_score','phat_score','slim_score_wt_mut','slim_score_mut_wt',]

    energy_features = ['wt_el_en','wt_el_en_net','wt_vdw_en','wt_vdw_en_net','wt_hb_en','wt_hb_en_net',
                          'wt_entropy_en','wt_entropy_en_net','wt_total_en','wt_total_en_net','wt_solvation_en',
                          'wt_solvation_en_net','mut_el_en','mut_el_en_net','mut_vdw_en','mut_vdw_en_net',
                          'mut_hb_en','mut_hb_en_net','mut_entropy_en','mut_entropy_en_net','mut_total_en',
                          'mut_total_en_net','mut_solvation_en','mut_solvation_en_net',]

    structure_features = ['wt_ss_H','wt_ss_G','wt_ss_I','wt_ss_E','wt_ss_B','wt_ss_C','wt_asa_abs','wt_asa_max',
                       'wt_asa_rel','wt_burried','wt_middle','wt_exposed','wt_contact_area','wt_contact_strength',
                       'wt_packing_density','wt_sidechain_volume','wt_nhb','wt_ncontacts_polar','wt_ncontacts_charged',
                       'wt_ncontacts_aliphatic','wt_ncontacts_aromatic','wt_ncontacts_special','mut_ss_H',
                       'mut_ss_G','mut_ss_I','mut_ss_E','mut_ss_B','mut_ss_C','mut_asa_abs','mut_asa_max','mut_asa_rel',
                       'mut_burried','mut_middle','mut_exposed','mut_contact_area','mut_contact_strength',
                       'mut_packing_density','mut_sidechain_volume','mut_nhb','mut_ncontacts_polar',
                       'mut_ncontacts_charged','mut_ncontacts_aliphatic','mut_ncontacts_aromatic',
                       'mut_ncontacts_special',]

    if feature_set=="all":
        return mutations_data, mutations_data_without_class

    elif feature_set=="custom":
        important_features = ['slim_score_mut_wt','mut_contact_strength','mut_total_en_net','mut_packing_density',
                              'wt_contact_area','mut_solvation_en','wt_sidechain_volume','wt_el_en','mut_contact_area',
                              'wt_total_en_net','blosum_score','wt_entropy_en_net','mut_isoelectric_point','mut_composition_membrane',
                              'wt_packing_density','wt_solvation_en_net','wt_vdw_en','wt_el_en_net','wt_asa_rel',
                              'mut_hb_en_net', 'mut_ncontacts_special', 'mut_entropy_en','mut_hydrophobicity_scale',
                              'mut_contribution_stability','mut_asa_rel','mut_nhb','mut_asa_abs','wt_vdw_en_net',
                              'wt_ncontacts_aromatic','mut_polarity','wt_ncontacts_aliphatic','mut_vdw_en',
                              'mut_vdw_en_net','wt_total_en','mut_el_en_net','mut_ncontacts_aliphatic','mut_sidechain_volume',
                              'wt_isoelectric_point','wt_entropy_en','wt_ncontacts_charged','wt_ncontacts_polar',]

        # drop all non-important features
        for feature in all_features:
            if feature not in important_features:
                mutations_data = mutations_data.drop(feature, axis=1)
                mutations_data_without_class = mutations_data_without_class.drop(feature, axis=1)

    elif feature_set=="sequence":
        for feature in all_features:
            if feature not in sequence_features:
                mutations_data = mutations_data.drop(feature, axis=1)
                mutations_data_without_class = mutations_data_without_class.drop(feature, axis=1)

    elif feature_set=="structure":
        for feature in all_features:
            if feature not in structure_features:
                mutations_data = mutations_data.drop(feature, axis=1)
                mutations_data_without_class = mutations_data_without_class.drop(feature, axis=1)

    elif feature_set=="energy":
        for feature in all_features:
            if feature not in energy_features:
                mutations_data = mutations_data.drop(feature, axis=1)
                mutations_data_without_class = mutations_data_without_class.drop(feature, axis=1)

    else:
        print "Unknown feature_set option", feature_set
        raise ValueError()

    return mutations_data, mutations_data_without_class


def getClassifier(mutations_data, mutations_data_without_class, n_estimators = 1000, train_test_ratio=0.1, model_name="model_tmp.pickle.dat", n_kfolds=20, n_splits=5):

    '''
    Here I derive the prediction model
    :return:
    '''
    scaler = StandardScaler()
    scaler.fit(mutations_data_without_class)
    scaled_features = pd.DataFrame(data=scaler.transform(mutations_data_without_class),
                                   columns=mutations_data_without_class.columns)

    X = scaled_features
    y = mutations_data[['Diseased', 'Benign']]

    X = X.as_matrix()
    y = y.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio, random_state=42,)

    print X_test

    y_train_label = []
    for el in y_train:
        y_train_label.append(el[0])

    print "There are %d disease-accosiated and %d benign mutations in the training set" % ( y_train_label.count(1.0),  y_train_label.count(0.0))

    w_disease = y_train_label.count(0.0)*1.0/len(y_train_label)
    w_benign = y_train_label.count(1.0)*1.0/len(y_train_label)

    weights_train = []
    for label in y_train_label:
        if label==1.0: weights_train.append(w_disease)
        else : weights_train.append(w_benign)

    y_test_label = []
    for el in y_test:
        y_test_label.append(el[0])

    print "There are %d disease-accosiated and %d benign mutations in the test set" % ( y_test_label.count(1.0),  y_test_label.count(0.0))

    scores_baseline = np.array([])
    n_estimators_baseline = n_estimators

    print 'Evaluating baseline model with %d n_estimators:' % (n_estimators_baseline)
    model_baseline = xgb.XGBClassifier(n_estimators=n_estimators_baseline)
    model_baseline.fit(X_train, y_train_label, sample_weight = weights_train)

    pickle.dump(model_baseline, open(model_name, "wb"))
    loaded_model = pickle.load(open(model_name, "rb"))

    print "###\nTRAININIG SET"
    y_predictions = model_baseline.predict(X_train)
    test_score_baseline = getScores(y_train_label, y_predictions)
    tn, fp, fn, tp = confusion_matrix(y_train_label, y_predictions).ravel()
    s = (tn+fp+fn+tp)/100.0
    print "tn, fp, fn, tp:", (tn, fp, fn, tp)
    print "tn, fp, fn, tp:", (tn/s, fp/s, fn/s, tp/s)

    if len(y_test_label)==0: return model_baseline

    print "###\nTEST SET"
    y_predictions = model_baseline.predict(X_test)
    test_score_baseline = getScores(y_test_label, y_predictions)
    tn, fp, fn, tp = confusion_matrix(y_test_label, y_predictions).ravel()
    s = (tn+fp+fn+tp)/100.0
    print "tn, fp, fn, tp:", (tn, fp, fn, tp)
    print "tn, fp, fn, tp:", (tn/s, fp/s, fn/s, tp/s)
    print "###"

    return model_baseline

def runCrossValidation(mutations_data, mutations_data_without_class, n_estimators = 50,
                       n_estimators_min=1, n_estimators_max=100, train_test_ratio=0.1,
                       metric='f1', plot=True):

    scaler = StandardScaler()
    scaler.fit(mutations_data_without_class)
    scaled_features = pd.DataFrame(data=scaler.transform(mutations_data_without_class),
                                   columns=mutations_data_without_class.columns)

    X = scaled_features
    y = mutations_data[['Diseased', 'Benign']]
    X = X.as_matrix()  # Convert X and y to Numpy arrays
    y = y.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio, random_state=42)

    y_train_label = []
    for el in y_train:
        y_train_label.append(el[0])

    w_disease = y_train_label.count(0.0)*1.0/len(y_train_label)
    w_benign = y_train_label.count(1.0)*1.0/len(y_train_label)
    weights_train = []
    for label in y_train_label:
        if label==1.0: weights_train.append(w_disease)
        else : weights_train.append(w_benign)

    y_test_label = []
    for el in y_test:
        y_test_label.append(el[0])

    n_kfolds = 20
    n_splits = 5

    scores_baseline = np.array([])
    n_estimators_baseline = n_estimators

    print 'Evaluating baseline model with %d n_estimators:' % (n_estimators_baseline)

    model_baseline = xgb.XGBClassifier(n_estimators=n_estimators_baseline)
    model_baseline.fit(X_train, y_train_label, sample_weight=weights_train)

    print "###\nTRAININIG SET"
    y_predictions = model_baseline.predict(X_train)
    test_score_baseline = getScores(y_train_label, y_predictions)
    print "###\nTEST SET"
    y_predictions = model_baseline.predict(X_test)
    test_score_baseline = getScores(y_test_label, y_predictions)
    print "###"

    print 'Running %d %d-fold validation for baseline model...' % (n_kfolds, n_splits)
    for i in range(n_kfolds):
        print i
        fold = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        scores_baseline_on_this_split = cross_val_score(estimator=xgb.XGBClassifier(n_estimators=n_estimators_baseline),
                                                        X=X_train, y=y_train_label,
                                                        cv=fold, scoring=metric,
                                                        fit_params = {'sample_weight':weights_train},
                                                        )
        scores_baseline = np.append(scores_baseline,
                                    scores_baseline_on_this_split)

    t_stats = []
    p_values = []
    n_trees = []
    test_scores = []

    print 'Running cross-validation for %d to %d estimators...' % (n_estimators_min, n_estimators_max)
    for j in range(n_estimators_min, n_estimators_max + 1):

        print 'Evaluating model with %d n_estimators...' % (j)
        model = xgb.XGBClassifier(n_estimators=j)
        model.fit(X_train, y_train_label)
        y_predictions = model.predict(X_test)
        test_score = getScores(y_test_label, y_predictions)
        test_scores.append(test_score)

        current_score = np.array([])
        for i in range(n_kfolds):
            print 'n_estimators : %3d; split : %3d ' % (j, i)
            fold = KFold(n_splits=n_splits, shuffle=True, random_state=i)
            scores_on_this_split = cross_val_score( estimator=xgb.XGBClassifier(n_estimators=j),
                                                    X=X_train, y=y_train_label,
                                                    cv=fold, scoring=metric,
                                                    fit_params={'sample_weight': weights_train},
                                                    )
            current_score = np.append(current_score, scores_on_this_split)

        t_stat, p_value = ttest_rel(current_score, scores_baseline)
        print 't_stat : %8.3f' % (t_stat)
        print 'p_value: %f' % (p_value)
        t_stats.append(t_stat)
        p_values.append(p_value)
        n_trees.append(j)

    if plot==True:
        plt.subplot(311)
        plt.plot(n_trees, t_stats)
        plt.xlabel('n_estimators')
        plt.ylabel('t-statistic')

        plt.subplot(312)
        plt.plot(n_trees, p_values)
        plt.xlabel('n_estimators')
        plt.ylabel('p-value w.r.t the baseline model')

        plt.subplot(313)
        plt.plot(n_trees, test_scores)
        plt.xlabel('n_estimators')
        plt.ylabel('scores')
        plt.legend()

        plt.show()

    return

def makePrediction(data, data_without_class, model):

    scaler = StandardScaler()
    scaler.fit(data_without_class)
    scaled_features = pd.DataFrame(data=scaler.transform(data_without_class),
                                   columns=data_without_class.columns)
    X = scaled_features
    y = data[['Diseased', 'Benign']]
    X = X.as_matrix()  # Convert X and y to Numpy arrays
    y = y.as_matrix()

    y_label = []
    for el in y:
        y_label.append(el[0])

    print "There are %d disease-accosiated and %d benign mutations" % (y_label.count(1.0), y_label.count(0.0))

    print "###\nRUNNING PREDICTIONS"
    y_predictions = model.predict(X)
    scores = getScores(y_label, y_predictions)
    if len(confusion_matrix(y_label, y_predictions).ravel())!=4:
        print "WARNING!!! BUG IN CONFUSION MATRIX"
        return scores, y_label, y_predictions

    tn, fp, fn, tp = confusion_matrix(y_label, y_predictions).ravel()
    s = (tn+fp+fn+tp)/100.0
    print "tn, fp, fn, tp:", (tn, fp, fn, tp)
    print "tn, fp, fn, tp:", (tn/s, fp/s, fn/s, tp/s)
    print "###"

    return scores, y_label, y_predictions

if __name__=="__main__":
    fn = "../resource/train.csv"
    data = pd.read_csv(fn)
    mutations_data, mutations_data_without_class = processData(data, feature_set = "all")
    print mutations_data.shape, mutations_data_without_class.shape

    flag_cross_validation = False
    if flag_cross_validation == True:
        runCrossValidation(mutations_data, mutations_data_without_class, 50, 1, 100, 0.1, 'f1', False)

    model = getClassifier(mutations_data, mutations_data_without_class, 77, 0.0)
    test_score_baseline, y_label, y_predictions = makePrediction(mutations_data, mutations_data_without_class, model)

    flag_save_model = False
    if flag_save_model:
        model_name = "model_pdbtm_77.pckl"
        pickle.dump(model, open(model_name, "wb"))
        loaded_model = pickle.load(open(model_name, "rb"))
