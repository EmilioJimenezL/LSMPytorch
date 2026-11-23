"""Re-export 3D CNN from models package (backward compatibility)."""

from models import count_parameters
from models.cnn3d import LSM3DCNNModel as LSM_3DCNN, create_3dcnn

__all__ = ["LSM_3DCNN", "create_3dcnn", "count_parameters"]
