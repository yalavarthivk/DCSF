"""Definitions of possible tasks."""
import abc

import asyncts.evaluation_metrics as metrics_module


class Task(abc.ABC):
    def __init__(self, class_weights=None):
        self._class_weights = class_weights

    @property
    def class_weights(self):
        return self._class_weights

    @property
    @abc.abstractmethod
    def loss(self):
        pass

    @property
    @abc.abstractmethod
    def output_activation(self):
        pass

    @property
    @abc.abstractmethod
    def n_outputs(self):
        pass

    @property
    @abc.abstractmethod
    def metrics(self):
        pass

    @property
    @abc.abstractmethod
    def monitor_quantity(self):
        pass

    @property
    @abc.abstractproperty
    def direction_of_improvement(self):
        pass

class BalanceBinaryClassification(Task):
    @property
    def n_classes(self):
        return 2

    @property
    def loss(self):
        return 'binary_crossentropy'

    @property
    def output_activation(self):
        return 'sigmoid'

    @property
    def n_outputs(self):
        return 1

    @property
    def metrics(self):
        return {
            'accuracy': metrics_module.accuracy
        }

    @property
    def monitor_quantity(self):
        return 'accuracy'

    @property
    def direction_of_improvement(self):
        return 'max'

class BinaryClassification(Task):
    @property
    def n_classes(self):
        return 2

    @property
    def loss(self):
        return 'binary_crossentropy'

    @property
    def output_activation(self):
        return 'sigmoid'

    @property
    def n_outputs(self):
        return 1

    @property
    def metrics(self):
        # TODO: Extend by further metrics
        return {
            'auprc': metrics_module.auprc,
            'auroc': metrics_module.auroc,
            'accuracy': metrics_module.accuracy
        }

    @property
    def monitor_quantity(self):
        return 'auroc'

    @property
    def direction_of_improvement(self):
        return 'max'
        # return 'min'
class OnlineMulticlassClassification(Task):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)
    @property
    def loss(self):
        return 'categorical_crossentropy'

    @property
    def output_activation(self):
        return 'softmax'

    @property
    def n_outputs(self):
        return (None, self.n_classes)

    @property
    def metrics(self):
        return {
            'accuracy': metrics_module.accuracy,
        }

    @property
    def monitor_quantity(self):
        return 'accuracy'

    @property
    def direction_of_improvement(self):
        return 'max'


class MulticlassClassification_ib(Task):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)

    @property
    def loss(self):
        return 'categorical_crossentropy'

    @property
    def output_activation(self):
        return 'softmax'

    @property
    def n_outputs(self):
        return self.n_classes

    @property
    def metrics(self):
        return {
            # 'auprc_micro': metrics_module.auprc_micro,
            # 'auprc_macro': metrics_module.auprc_macro,
            # 'auprc_weighted': metrics_module.auprc_weighted,
            # 'auroc_micro': metrics_module.auroc_micro,
            # 'auroc_macro': metrics_module.auroc_macro,
            # 'auroc_weighted': metrics_module.auroc_weighted,
            'accuracy': metrics_module.accuracy
        }

    @property
    def monitor_quantity(self):
        return 'loss'

    @property
    def direction_of_improvement(self):
        return 'min'


class MulticlassClassification(Task):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)

    @property
    def loss(self):
        return 'categorical_crossentropy'

    @property
    def output_activation(self):
        return 'softmax'

    @property
    def n_outputs(self):
        return self.n_classes

    @property
    def metrics(self):
        return {
            'accuracy': metrics_module.accuracy
        }

    @property
    def monitor_quantity(self):
        return 'accuracy'

    @property
    def direction_of_improvement(self):
        return 'max'


class MultilabelClassification(Task):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)

    @property
    def loss(self):
        return 'binary_crossentropy'

    @property
    def output_activation(self):
        return 'sigmoid'

    @property
    def n_outputs(self):
        return self.n_classes

    @property
    def metrics(self):
        return {
            'auprc_micro': metrics_module.auprc_micro,
            'auprc_macro': metrics_module.auprc_macro,
            'auprc_weighted': metrics_module.auprc_weighted,
            'auroc_micro': metrics_module.auroc_micro,
            'auroc_macro': metrics_module.auroc_macro,
            'auroc_weighted': metrics_module.auroc_weighted,
        }

    @property
    def monitor_quantity(self):
        return 'auprc_weighted'

    @property
    def direction_of_improvement(self):
        return 'max'


class Regression(Task):
    def __init__(self, n_dimensions, is_positive):
        self.n_dimensions = n_dimensions
        self.is_positive = is_positive

    @property
    def loss(self):
        return 'mean_squared_logarithmic_error'

    @property
    def output_activation(self):
        return 'relu' if self.is_positive else 'linear'

    @property
    def n_outputs(self):
        return self.n_dimensions

    @property
    def metrics(self):
        return {
            'mean_absolute_error': metrics_module.mean_absolute_error,
            'mean_squared_error': metrics_module.mean_squared_error,
            'mean_absolute_percentage_error':
            metrics_module.mean_absolute_percentage_error
        }

    @property
    def monitor_quantity(self):
        return 'loss'

    @property
    def direction_of_improvement(self):
        return 'min'


DATASET_TO_TASK_MAPPING = {
    'physionet2012': BinaryClassification(),
    'miniphysionet':BinaryClassification(),
    'mimic3_mortality': BinaryClassification(),
    'LSST_0_1':MulticlassClassification(14),
    'LSST_0_5':MulticlassClassification(14),
    'LSST_0_9':MulticlassClassification(14),
    'LSST_async':MulticlassClassification(14),
    'PS_0_1':MulticlassClassification(39),
    'PS_0_5':MulticlassClassification(39),
    'PS_0_9':MulticlassClassification(39),
    'PS_async':MulticlassClassification(39),
    'Synth_0_5':BalanceBinaryClassification(),
    'Synth_0_1':BalanceBinaryClassification(),
    'Synth_0_9':BalanceBinaryClassification(),
    'activity':OnlineMulticlassClassification(11)
}
