"""Re-export 1D CNN from models package (backward compatibility)."""

from models import count_parameters
from models.cnn1d import CNN1D, LSM_CNN, create_cnn1d

__all__ = ["LSM_CNN", "CNN1D", "create_cnn1d", "count_parameters"]
