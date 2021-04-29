from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees import HoeffdingTree, HAT
from xgboost import XGBClassifier

from src.clustering.clustering import Clustering
from src.core.common import get_method_config
from src.encoding.models import Encoding
from src.jobs.models import Job, ModelType
from src.labelling.models import LabelTypes
from src.predictive_model.classification.custom_classification_models import NNClassifier
from src.predictive_model.classification.models import ClassificationMethods
from src.utils.django_orm import duplicate_orm_row
from src.utils.result_metrics import calculate_results_classification, get_auc

pd.options.mode.chained_assignment = None

import logging

logger = logging.getLogger(__name__)


def classification(training_df: DataFrame, test_df: DataFrame, clusterer: Clustering, job: Job) -> (dict, dict):
    """main classification entry point

    train and tests the classifier using the provided data

    :param clusterer:
    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data = _drop_columns(training_df)
    test_data = _drop_columns(test_df)

    #estraggo le labels di test
    labels_test = test_df['trace_id']

    # job.encoding = duplicate_orm_row(Encoding.objects.filter(pk=job.encoding.pk)[0])  # TODO: maybe here would be better an intelligent get_or_create...
    job.encoding = Encoding.objects.create(
        data_encoding=job.encoding.data_encoding,
        value_encoding=job.encoding.value_encoding,
        add_elapsed_time=job.encoding.add_elapsed_time,
        add_remaining_time=job.encoding.add_remaining_time,
        add_executed_events=job.encoding.add_executed_events,
        add_resources_used=job.encoding.add_resources_used,
        add_new_traces=job.encoding.add_new_traces,
        features=job.encoding.features,
        prefix_length=job.encoding.prefix_length,
        padding=job.encoding.padding,
        task_generation_type=job.encoding.task_generation_type
    )
    job.encoding.features = list(train_data.columns.values)
    job.encoding.save()
    job.save()

    model_split = _train(train_data, _choose_classifier(job), clusterer)
    empty_df = pd.DataFrame()

    results_df, auc = _test(
        empty_df,
        model_split,
        test_data,
        labels_test,
        evaluation=True,
        is_binary_classifier=_check_is_binary_classifier(job.labelling.type)
    )

    results = _prepare_results(results_df, auc)

    return results, model_split


def update_and_test(training_df: DataFrame, test_df: DataFrame, job: Job):
    train_data = _drop_columns(training_df)
    test_data = _drop_columns(test_df)

    job.encoding = job.incremental_train.encoding
    job.encoding.save()
    job.save()

    if list(train_data.columns.values) != job.incremental_train.encoding.features:
        # TODO: how do I align the two feature vectors?
        train_data, _ = train_data.align(
            pd.DataFrame(columns=job.incremental_train.encoding.features), axis=1, join='right')
        train_data = train_data.fillna(0)
        test_data, _ = test_data.align(
            pd.DataFrame(columns=job.incremental_train.encoding.features), axis=1, join='right')
        test_data = test_data.fillna(0)

    # TODO: UPDATE if incremental, otherwise just test
    model_split = _update(job, train_data)

    results_df, auc = _test(model_split, test_data, evaluation=True,
                            is_binary_classifier=_check_is_binary_classifier(job.labelling.type))

    results = _prepare_results(results_df, auc)

    return results, model_split


def _train(train_data: DataFrame, classifier: ClassifierMixin, clusterer: Clustering) -> dict:
    models = dict()

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):
        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            cluster_targets_df = DataFrame(cluster_train_df['label'])
            try:
                classifier.fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())
            except (NotImplementedError, KeyError):
                classifier.partial_fit(cluster_train_df.drop('label', 1).values, cluster_targets_df.values.ravel())
            except Exception as exception:
                raise exception

            models[cluster] = classifier
            try:
                classifier = clone(classifier)
            except TypeError:
                classifier = clone(classifier, safe=False)
                classifier.reset()

    return {ModelType.CLUSTERER.value: clusterer, ModelType.CLASSIFIER.value: models}


def _update(job: Job, data: DataFrame) -> dict:
    previous_job = job.incremental_train

    clusterer = Clustering.load_model(previous_job)

    update_data = clusterer.cluster_data(data)

    models = joblib.load(previous_job.predictive_model.model_path)
    if job.predictive_model.prediction_method in [ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                                                  ClassificationMethods.ADAPTIVE_TREE.value,
                                                  ClassificationMethods.HOEFFDING_TREE.value,
                                                  ClassificationMethods.SGDCLASSIFIER.value,
                                                  ClassificationMethods.PERCEPTRON.value,
                                                  ClassificationMethods.RANDOM_FOREST.value]:  # TODO: workaround
        print('entered update')
        for cluster in range(clusterer.n_clusters):
            x = update_data[cluster]
            if not x.empty:
                y = x['label']
                try:
                    if previous_job.predictive_model.prediction_method == ClassificationMethods.RANDOM_FOREST.value:
                        models[cluster].fit(x.drop('label', 1), y.values.ravel())
                    else:
                        models[cluster].partial_fit(x.drop('label', 1), y.values.ravel())
                except (NotImplementedError, KeyError):
                    if previous_job.predictive_model.prediction_method == ClassificationMethods.RANDOM_FOREST.value:
                        models[cluster].fit(x.drop('label', 1).values, y.values.ravel())
                    else:
                        models[cluster].partial_fit(x.drop('label', 1).values, y.values.ravel())
                except Exception as exception:
                    raise exception

    return {ModelType.CLUSTERER.value: clusterer, ModelType.CLASSIFIER.value: models}


def _test(remaining_test:DataFrame, model_split: dict, test_data: DataFrame,labels_traces, evaluation: bool,
          is_binary_classifier: bool) -> (
    DataFrame, float):
    clusterer = model_split[ModelType.CLUSTERER.value]
    classifier = model_split[ModelType.CLASSIFIER.value]

    test_data = clusterer.cluster_data(test_data)

    results_df = DataFrame()
    auc = 0

    non_empty_clusters = clusterer.n_clusters

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            cluster_targets_df = cluster_test_df['label']
            if evaluation:
                try:
                    if hasattr(classifier[cluster], 'decision_function'):
                        scores = classifier[cluster].decision_function(cluster_test_df.drop(['label'], 1))
                    else:


                        scores = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1))


                        #labels del classificatore
                        label_classes = classifier[cluster].classes_

                        scores_long = scores

                        if np.size(scores, 1) >= 2:  # checks number of columns
                            scores = scores[:, 1]
                except (NotImplementedError, KeyError):
                    if hasattr(classifier[cluster], 'decision_function'):
                        scores = classifier[cluster].decision_function(cluster_test_df.drop(['label'], 1).values)
                    else:
                        scores = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1).values)

                        try:
                            if np.size(scores, 1) >= 2:  # checks number of columns
                                scores = scores[:, 1]
                        except Exception as exception:
                            pass
                auc += get_auc(cluster_targets_df, scores)
            try:
                cluster_test_df['predicted'] = classifier[cluster].predict(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                cluster_test_df['predicted'] = classifier[cluster].predict(cluster_test_df.drop(['label'], 1).values)

            results_df = results_df.append(cluster_test_df)

    if is_binary_classifier or max([len(set(t['label'])) for _, t in test_data.items()]) <= 2:
        auc = float(auc) / non_empty_clusters
    else:
        pass  # TODO: check if AUC is ok for multiclass, otherwise implement

    ########

    # type(scores_long) numpy.array
    len_scores = np.size(scores_long, 0)  # prendo la dimensione di scores_long,la lunghezza
    index_traces = list(range(0, len_scores))  # creo una lista che va da 0 alla lunghezza di scores long



    # labels_traces sono le label dei miei test data
    labels_traces.index = index_traces  # modifico l'indice delle labels


    # predicted= pd.Series(cluster_test_df['predicited'],index=index_traces)


    #cambio gli indici
    #cluster_test_df sono i dati nel test che hanno un indice diverso
    cluster_test_df.index = index_traces  # predicited è ciò che
    #cluster_targets_df sono i risultati della prediction
    cluster_targets_df.index = index_traces

    #estraggo gli indixi delle prob massime che ottengo
    index_max_scores = np.argmax(scores_long,axis=1)  # index of the max prob. RICAVO GLI INDICI MASSIMI DI SCORES LONG


    #se ho l'indice di massima prob 29, 29 in label corrisponde ad altro

    #print("labels")
    #print(label_classes)

    #estraggo i giusti indici della massime probabilità
    #label_classes sono le label che ottengo dal modello e che devo cambiare
    label_max_index = label_classes[index_max_scores]  # right index , right labels

    # retrieve the positions of the 5 bigger probabilities in scores, Ricavo le prime 5 predizioni

    pos_prob_5 = position_probab_top5(scores_long)

    # retreive the associated RIGHT position in labels, ricavo le giuste posizioni associate alle labels in
    final_pos = position_labels(pos_prob_5, label_classes)

    top_5 = pd.Series(final_pos, index=index_traces)

    with open("/Users/biancaciuche/PycharmProjects/scores.csv", 'wb') as f:
        np.savetxt(f, scores_long, delimiter=",")

    if not (remaining_test.empty):
        cont = [ ]
        for index, row in remaining_test.iterrows():
            l = remaining_test.iloc[ index ].values
            cont.append(l)
        remain = pd.Series(cont)
        all_data = pd.DataFrame({'label': labels_traces, 'remainig': remain, 'top_1': label_max_index, 'top_5': top_5}, index=index_traces)



    if not(remaining_test.empty):
        precision_top1(remaining_test, all_data)
        precision_top5(remaining_test, all_data)
        all_data.to_csv('/Users/biancaciuche/PycharmProjects/results_1.csv')

    return results_df, auc

def precision_top1(remaining_test,all_data):
    TP_1 = 0
    presence = [ ]
    for index, row in all_data.iterrows():

        if (row[ 'top_1' ] in remaining_test.iloc[ index ].values):
            presence.append("TRUE")
            TP_1 += 1
        else:
            presence.append("FALSE")

    print('precision1')
    print((TP_1) / all_data.shape[0])
    all_data.insert(4, "Found_1", presence, True)
    return

def precision_top5(remaining_test,all_data):
    TP_5=0
    presence = [ ]
    for index, row in all_data.iterrows():
        c = 0
        for i in row[ 'top_5' ]:
            if (i in remaining_test.iloc[ index ].values):
                c += 1

        if c >= 1:
            presence.append("TRUE")
            TP_5 += c / 5
        else:
            presence.append("FALSE")

    print('precision5')
    print(TP_5 / all_data.shape[ 0 ])
    all_data.insert(5, "Found_5", presence, True)


def position_labels(pos_prob,label_classes):
    final_pos = []
    for i in pos_prob:

        new = [] #right labels
        for pos in i:
            new.append(label_classes[pos])
        final_pos.append(new)

    return final_pos


def position_probab_top5(scores):
    #scores è l'array che contiene tutte le prob
    L = []

    for i in scores:
        i = i.tolist()
        #print(i)
        top_5 = []
        dec_list= sorted(i,reverse=True)#ordine decrescente
        # delete duplicates
        dec_list= list(dict.fromkeys(dec_list))[0:-1]

        if dec_list[0]<=1: #se il primo elemento della riga che sto considerando è minore/uguale a 1
            count = 0 #conto il numero di elemnti che sto considerando
            for j in range(0,len(dec_list)):
                if count<=5:
                    #prendo l'elemento dec_list[j]

                    #devo capire il numero di elementi nella riga i che sono uguali a dec_list[j]
                    index_element = [t for t,x in enumerate(i) if x == dec_list[j]] #ricavo gli indici delle posizioni
                    count += len(index_element)

                    list.extend(top_5,index_element)
                    if len(top_5)>5:
                        top_5 = top_5[0:5]
        # top 5 hold the index of the 5 max probabilities
        else:
            list.extend(top_5,i.index(dec_list[0]))

        L.append(top_5)
    return L

def predict(job: Job, data: DataFrame) -> Any:
    data = data.drop(['trace_id'], 1)
    clusterer = Clustering.load_model(job)
    data = clusterer.cluster_data(data)

    classifier = joblib.load(job.predictive_model.model_path)

    non_empty_clusters = clusterer.n_clusters

    result = None

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            try:
                result = classifier[cluster].predict(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                result = classifier[cluster].predict(cluster_test_df.drop(['label'], 1).values)

    return result


def predict_proba(job: Job, data: DataFrame) -> Any:
    data = data.drop(['trace_id'], 1)
    clusterer = Clustering.load_model(job)
    data = clusterer.cluster_data(data)

    classifier = joblib.load(job.predictive_model.model_path)

    non_empty_clusters = clusterer.n_clusters

    result = None

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = data[cluster]
        if cluster_test_df.empty:
            non_empty_clusters -= 1
        else:
            try:
                result = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1))
            except (NotImplementedError, KeyError):
                result = classifier[cluster].predict_proba(cluster_test_df.drop(['label'], 1).values)

    return result


def _prepare_results(results_df: DataFrame, auc: int) -> dict:
    actual = results_df['label'].values
    predicted = results_df['predicted'].values

    row = calculate_results_classification(actual, predicted)
    row['auc'] = auc
    return row


def _drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop('trace_id', 1)
    return df


def _choose_classifier(job: Job):
    method, config = get_method_config(job)
    config.pop('classification_method', None)
    logger.info("Using method {} with config {}".format(method, config))
    if method == ClassificationMethods.KNN.value:
        classifier = KNeighborsClassifier(**config)
    elif method == ClassificationMethods.RANDOM_FOREST.value:
        classifier = RandomForestClassifier(**config)
    elif method == ClassificationMethods.DECISION_TREE.value:
        classifier = DecisionTreeClassifier(**config)
    elif method == ClassificationMethods.XGBOOST.value:
        classifier = XGBClassifier(**config)
    elif method == ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value:
        classifier = MultinomialNB(**config)
    elif method == ClassificationMethods.ADAPTIVE_TREE.value:
        classifier = HAT(**config)
    elif method == ClassificationMethods.HOEFFDING_TREE.value:
        classifier = HoeffdingTree(**config)
    elif method == ClassificationMethods.SGDCLASSIFIER.value:
        classifier = SGDClassifier(**config)
    elif method == ClassificationMethods.PERCEPTRON.value:
        classifier = Perceptron(**config)
    elif method == ClassificationMethods.NN.value:
        config['encoding'] = job.encoding.value_encoding
        config['is_binary_classifier'] = _check_is_binary_classifier(job.labelling.type)
        classifier = NNClassifier(**config)
    else:
        raise ValueError("Unexpected classification method {}".format(method))
    return classifier


def _check_is_binary_classifier(label_type: str) -> bool:
    if label_type in [LabelTypes.REMAINING_TIME.value, LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value]:
        return True
    if label_type in [LabelTypes.NEXT_ACTIVITY.value, LabelTypes.ATTRIBUTE_STRING.value]:
        return False
    raise ValueError("Label type {} not supported".format(label_type))
