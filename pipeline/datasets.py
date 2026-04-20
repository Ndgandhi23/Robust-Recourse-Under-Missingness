"""
Dataset registry.

Each dataset is a lazy callable that returns a DatasetConfig.
Nothing is loaded or downloaded until get_dataset() is called.
"""

from .dataset_config import DatasetConfig

# lazy imports — avoids pulling in pandas / download logic at import time
_REGISTRY = {}


def _register(name, module_path, config_fn_name="get_config"):
    """Register a dataset by module path (imported on first access)."""
    _REGISTRY[name] = (module_path, config_fn_name)


# ── registered datasets ──────────────────────────────────────────────
_register("diabetes",   "pipeline.data_diabetes")
_register("fico",       "pipeline.data_fico")


def get_dataset(name: str) -> DatasetConfig:
    """Load and return a DatasetConfig by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    module_path, fn_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name)()


def list_datasets():
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY.keys())
