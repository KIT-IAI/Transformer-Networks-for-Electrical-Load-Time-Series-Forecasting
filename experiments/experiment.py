import datetime
import json
import os

from evaluation.evaluator import Evaluation
from models.wrappers.base_model_wrapper import BaseModelWrapper
from training.trainer import TrainingReport
from training.training_config import TrainingConfig

JSON_FILE_ENDING = '.json'


class Experiment:
    """
    Collects all important information needed for reproducing results of trained model.
    """

    def __init__(self, model_wrapper: BaseModelWrapper, evaluation: Evaluation, training_config: TrainingConfig,
                 training_report: TrainingReport):
        self.model_wrapper = model_wrapper
        self.evaluation = evaluation
        self.training_config = training_config
        self.training_report = training_report

    def save_to_json_file(self) -> None:
        result = {
            'modelType': str(self.model_wrapper.model_type),
            'modelWrapper': str(self.model_wrapper),
            'trainingConfig': self.training_config.__dict__,
            'trainingReport': self.training_report.serialize(),
            'evaluation': self.evaluation.__dict__
        }
        date = str(datetime.datetime.now()) \
            .replace(' ', '_') \
            .replace('-', '_') \
            .replace(':', '_') \
            .replace('.', '_')
        experiment_name = str(self.model_wrapper.model_type) + date
        file_path = os.path.join('experiments', 'archive', experiment_name + JSON_FILE_ENDING)

        with open(file_path, 'w') as fp:
            json.dump(result, fp)
            fp.close()

    def __str__(self):
        return 'Model type: ' + str(self.model_wrapper.model_type) + '\n' \
               + 'Model architecture: ' + str(self.model_wrapper) + '\n' \
               + 'Training configuration: ' + str(self.training_config) + '\n' \
               + 'Training report: ' + str(self.training_report) + '\n' \
               + 'Evaluation: ' + str(self.evaluation)
