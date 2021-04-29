import django
django.setup()

from src.predictive_model.classification.models import ClassificationMethods
from src.encoding.models import TaskGenerationTypes, ValueEncodings
from src.hyperparameter_optimization.models import HyperparameterOptimizationMethods, HyperOptLosses, HyperOptAlgorithms
from src.labelling.models import LabelTypes
from src.utils.experiments_utils import create_classification_payload, send_job_request, upload_split


split_id = upload_split(
    #train='/Users/biancaciuche/PycharmProjects/recommandation/data/Florence/trajectories_florence_extended_training
    # .xes',
    #test='/Users/biancaciuche/PycharmProjects/recommandation/data/Florence/trajectories_florence_extended_testing.xes',
    train='/Users/biancaciuche/PycharmProjects/recommandation/data/Rome/trajectories_rome_extended_training.xes',
    test='/Users/biancaciuche/PycharmProjects/recommandation/data/Rome/trajectories_rome_extended_testing.xes',
    #train='/Users/biancaciuche/PycharmProjects/recommandation/sepsis/Sepsis Cases - Event Log_training.xes',
    #test='/Users/biancaciuche/PycharmProjects/recommandation/sepsis/Sepsis Cases - Event Log_testing.xes',
    server_name='localhost',
    server_port='8000',
    #train_name= ('path_nome_file_train.xes').replace('/', '_'),
    #test_name= ('path_nome_file_test.xes').replace('/', '_')
)
print(split_id)

# 12->rome'''


#24 ->sepsis
#25 ->Florence
job = send_job_request(
    payload=create_classification_payload(
        split=split_id,
        encodings=[ValueEncodings.COMPLEX.value],
        encoding={"padding": "zero_padding",
                  "generation_type": TaskGenerationTypes.ONLY_THIS.value, #ONLY_THIS
                  "prefix_length":0.7, #50,
                  "features": []},
        labeling={"type": LabelTypes.NEXT_ACTIVITY.value,
                  "attribute_name": "label",
                  "add_remaining_time": False,
                  "add_elapsed_time": False,
                  "add_executed_events": False,
                  "add_resources_used": False,
                  "add_new_traces": False},
        hyperparameter_optimization={"type": HyperparameterOptimizationMethods.HYPEROPT.value,
                                     "max_evaluations": 5,
                                     "performance_metric": HyperOptLosses.F1SCORE.value,
                                     "algorithm_type": HyperOptAlgorithms.TPE.value},
        classification=[ClassificationMethods.KNN.value,
                        #ClassificationMethods.DECISION_TREE.value,
                        #ClassificationMethods.RANDOM_FOREST.value,
                        #ClassificationMethods.XGBOOST.value,
                        #ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value,
                        #ClassificationMethods.SGDCLASSIFIER.value,
                        #ClassificationMethods.PERCEPTRON.value,
        ]
    ), server_name='localhost', server_port='8000'
)[0]['id']

print(job)
