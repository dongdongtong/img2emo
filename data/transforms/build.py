from utils.register import Registry
from utils.tools import check_availability

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transform(cfg, is_train=False, only_other_transforms=False):
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    check_availability(cfg.INPUT.TRANSFORMS, avai_transforms)
    if cfg.VERBOSE:
        print("Loading transforms: {}".format(cfg.INPUT.TRANSFORMS))
    return TRANSFORM_REGISTRY.get(cfg.INPUT.TRANSFORMS)(cfg, is_train, only_other_transforms)


def build_transform_from_name(cfg, transform_name, is_train=False, only_other_transforms=False):
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    check_availability(transform_name, avai_transforms)
    if cfg.VERBOSE:
        print("Loading transforms: {}".format(transform_name))
    return TRANSFORM_REGISTRY.get(transform_name)(cfg, is_train, only_other_transforms)
