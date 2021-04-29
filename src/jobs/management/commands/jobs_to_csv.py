import csv

from django.core.management.base import BaseCommand

from src.evaluation.models import Evaluation
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'helps requeue properly jobs that have been remove from both default and failed queue in redis'

    def handle(self, *args, **kwargs):
        jobs = [j for j in Job.objects.all() if j.status in ['completed', 'running'] and j.id >= 312]
        jobs_dict = [j.to_dict() for j in jobs if j.status in ['completed', 'running'] and j.id >= 312]

        header, data = jobs_to_proper_format(jobs, jobs_dict)

        with open('DUMP_RESULTS.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(header)
            writer.writerows(data)

        writeFile.close()

        print('done')


def jobs_to_proper_format(jobs, jobs_dict):
    for j1, j2 in zip(jobs_dict, jobs):
        try:
            j1['evaluation'] = Evaluation.objects.filter(pk=j2.evaluation.pk).select_subclasses()[0].get_full_dict()
        except:
            pass
        try:
            j1['predictive_model']['id'] = j2.predictive_model.id
        except:
            pass
        try:
            j1['id'] = j2.id
        except:
            pass
        try:
            j1['hyperparameter_optimizer'] = \
            HyperparameterOptimization.objects.filter(pk=j2.hyperparameter_optimizer.pk).select_subclasses()[
                0].get_full_dict()
        except:
            pass

    header = [['split_id',
               'predictive_model_id',
               'incremental_model_id',
               'job_id',

               'predictive_model',
               'evaluation_auc',
               'evaluation_f1_score',
               'evaluation_accuracy',
               'evaluation_precision',
               'evaluation_recall',
               'evaluation_elapsed_time',
               'hyperparameter_optimizer_elapsed_time',

               'evaluation_true_positive',
               'evaluation_false_positive',
               'evaluation_true_negative',
               'evaluation_false_negative',

               'hyperparameter_optimizer_performance_metric',
               'hyperparameter_optimizer_max_evaluations',

               'encoding_value_encoding',
               'encoding_prefix_length',
               'encoding_task_generation_type',
               'labelling_type',
               'labelling_attribute_name',
               'clustering_method',

               ]]
    data = [[
        j['split']['id'] if 'id' in j['split'] else '',
        j['predictive_model']['id'] if 'id' in j['predictive_model'] else '',
        j['incremental_train'][0]['id'] if j['incremental_train'][0] is not None else '',
        j['id'] if 'id' in j else '',

        j['predictive_model']['prediction_method'] if 'prediction_method' in j['predictive_model'] else '',
        j['evaluation']['auc'] if 'auc' in j['evaluation'] else '',
        j['evaluation']['f1_score'] if 'f1_score' in j['evaluation'] else '',
        j['evaluation']['accuracy'] if 'accuracy' in j['evaluation'] else '',
        j['evaluation']['precision'] if 'precision' in j['evaluation'] else '',
        j['evaluation']['recall'] if 'recall' in j['evaluation'] else '',
        j['evaluation']['elapsed_time'] if 'elapsed_time' in j['evaluation'] else '',
        j['hyperparameter_optimizer']['elapsed_time'] if 'elapsed_time' in j['hyperparameter_optimizer'] else '',

        j['evaluation']['true_positive'] if 'true_positive' in j['evaluation'] else '',
        j['evaluation']['false_positive'] if 'false_positive' in j['evaluation'] else '',
        j['evaluation']['true_negative'] if 'true_negative' in j['evaluation'] else '',
        j['evaluation']['false_negative'] if 'false_negative' in j['evaluation'] else '',

        j['hyperparameter_optimizer']['performance_metric'] if 'performance_metric' in j['hyperparameter_optimizer'] else '',
        j['hyperparameter_optimizer']['max_evaluations'] if 'max_evaluations' in j['hyperparameter_optimizer'] else '',

        j['encoding']['value_encoding'] if 'value_encoding' in j['encoding'] else '',
        j['encoding']['prefix_length'] if 'prefix_length' in j['encoding'] else '',
        j['encoding']['task_generation_type'] if 'task_generation_type' in j['encoding'] else '',
        j['labelling']['type'] if 'type' in j['labelling'] else '',
        j['labelling']['attribute_name'] if 'attribute_name' in j['labelling'] else '',
        j['clustering']['clustering_method'] if 'clustering_method' in j['clustering'] else '',

    ] for j in jobs_dict]

    return header, data
