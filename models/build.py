from utils.register import Registry
from utils.tools import check_availability

MODEL_REGISTRY = Registry("MODEL")


def build_model(cfg):
    avai_models = MODEL_REGISTRY.registered_names()
    check_availability(cfg.MODEL.NAME, avai_models)
    if cfg.VERBOSE:
        print("Loading model: {}".format(cfg.MODEL.NAME))
    return MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)