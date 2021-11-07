class TrainingConfig:

    def __init__(self, learning_rate: float, max_epochs: int, use_early_stopping: bool, early_stopping_patience: int):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience

    def __str__(self):
        return str(self.__dict__)
