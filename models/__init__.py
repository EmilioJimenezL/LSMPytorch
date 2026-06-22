from .cnn1d import CNN1D, LSM_CNN, create_cnn1d
from .tcn   import TemporalConvNet, create_tcn
from .cnn3d import LSM3DCNNModel, create_3dcnn

MODEL_REGISTRY = {
    '1d_cnn': create_cnn1d,
    'tcn':    create_tcn,
    '3d_cnn': create_3dcnn,
}

# CLI names in train.py → registry keys
TRAIN_MODEL_MAP = {
    'cnn':   '1d_cnn',
    'tcn':   'tcn',
    '3dcnn': '3d_cnn',
}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

MODEL_INFO = {
    '1d_cnn': {
        'name': '1D CNN (Baseline)',
        'parameters': '~0.76M',
        'size_mb': '3-4',
        'training_time_h': '13-17',
        'accuracy': '80-85%',
        'speed_ms': '15-20',
    },
    'tcn': {
        'name': 'Temporal Convolutional Network',
        'parameters': '~1.8M',
        'size_mb': '7-10',
        'training_time_h': '10-12',
        'accuracy': '85-87%',
        'speed_ms': '100-150',
    },
    '3d_cnn': {
        'name': '3D CNN with Temporal Attention',
        'parameters': '~0.2M',
        'size_mb': '1-2',
        'training_time_h': '12-15',
        'accuracy': '80-85%',
        'speed_ms': '80-90',
    },
}


def create_model(model_name: str, num_classes: int, **kwargs):
    """
    Factory function for creating models.

    Args:
        model_name: '1d_cnn', 'tcn', or '3d_cnn'
        num_classes: Number of output classes
        **kwargs: Model-specific arguments (dropout, etc.)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](num_classes, **kwargs)


def get_model_info(model_name: str) -> dict:
    return MODEL_INFO.get(model_name, {})


__all__ = [
    'create_model',
    'get_model_info',
    'MODEL_REGISTRY',
    'TRAIN_MODEL_MAP',
    'MODEL_INFO',
    'count_parameters',
    'CNN1D',
    'LSM_CNN',
    'TemporalConvNet',
    'LSM3DCNNModel',
]
