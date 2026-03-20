# training.py


class ModelTrainer:
    def __init__(self, model_cls, model_params=None):
        self.model_cls = model_cls
        self.model_params = model_params or {}

    def train(self, X, y):
        model = self.model_cls(**self.model_params)
        model.fit(X, y)
        return model


class MultiTargetTrainer:
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def train_all(self, X, y_dict):
        models = {}
        for target, y in y_dict.items():
            models[target] = self.model_trainer.train(X, y)
        return models
