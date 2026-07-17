"""Re-export TCN from models package (backward compatibility)."""

from models.tcn import TemporalConvNet as LSM_TCN, create_tcn

__all__ = ["LSM_TCN", "create_tcn"]
