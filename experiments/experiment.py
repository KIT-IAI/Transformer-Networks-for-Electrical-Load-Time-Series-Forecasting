import argparse
import datetime
import json
import os

from evaluation.evaluator import Evaluation
from models.wrappers.base_model_wrapper import BaseModelWrapper
from training.trainer import TrainingReport

JSON_FILE_ENDING = '.json'
EXPERIMENTS_DIRECTORY = 'experiments'
FINAL_EXPERIMENTS_DIRECTORY = '2022-09-20'


class Experiment:
    """
    Collects all important information needed for reproducing and analysing results of trained model.
    """

    def __init__(self, model_wrapper: BaseModelWrapper, evaluation: Evaluation, training_config: argparse.Namespace,
                 training_report: TrainingReport, training_time: float, test_time: float):
        self.model_wrapper = model_wrapper
        self.evaluation = evaluation
        self.training_config = training_config
        self.training_report = training_report
        self.training_time = training_time
        self.test_time = test_time

    def save_to_json_file(self) -> None:
        """
        Saves the experiment data to a json file. The name of the file is specified by the executed model and the time
        of execution.
        The data is serialized before storing.
        """

        date = str(datetime.datetime.now()) \
            .replace(' ', '_') \
            .replace('-', '_') \
            .replace(':', '_') \
            .replace('.', '_')
        experiment_name = str(self.model_wrapper.model_type) + date
        print(experiment_name)

        serialized_training_report = self.training_report.serialize() if self.training_report else None
        result = {
            'experimentName': experiment_name,
            'modelType': str(self.model_wrapper.model_type),
            'modelWrapper': str(self.model_wrapper),
            'trainingConfig': self.training_config.__dict__,
            'trainingReport': serialized_training_report,
            'evaluation': self.evaluation.serialize(),
            'training_time': self.training_time,
            'test_time': self.test_time
        }

        file_path = os.path.join(EXPERIMENTS_DIRECTORY, FINAL_EXPERIMENTS_DIRECTORY, experiment_name + JSON_FILE_ENDING)

        with open(file_path, 'w') as fp:
            json.dump(result, fp)
            fp.close()

    def __str__(self):
        return 'Model type: ' + str(self.model_wrapper.model_type) + '\n' \
               + 'Model architecture: ' + str(self.model_wrapper) + '\n' \
               + 'Training configuration: ' + str(self.training_config) + '\n' \
               + 'Training report: ' + str(self.training_report) + '\n' \
               + 'Evaluation: ' + str(self.evaluation)
