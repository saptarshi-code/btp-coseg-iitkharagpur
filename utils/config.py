"""utils/config.py — YAML config loader with dot-access."""

import yaml
from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = SimpleNamespace(**d)
    # provide defaults for optional keys
    if not hasattr(cfg, 'conf_threshold'):
        cfg.conf_threshold = None
    if not hasattr(cfg, 'use_mas'):
        cfg.use_mas = False
    if not hasattr(cfg, 'use_ema'):
        cfg.use_ema = False
    cfg.get = lambda key, default=None: getattr(cfg, key, default)
    return cfg
