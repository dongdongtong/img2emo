from utils import Registry, check_availability


DATASET_REGISTRY = Registry('DATASET')
DATASET_WRAPPER_REGISTRY = Registry('DATASET_WRAPPER')


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)


def build_dataset_wrapper(cfg):
    avai_dataset_wrappers = DATASET_WRAPPER_REGISTRY.registered_names()
    check_availability(cfg.DATASET_WRAPPER.NAME, avai_dataset_wrappers)
    if cfg.VERBOSE:
        print("Loading dataset wrapper: {}".format(cfg.DATASET_WRAPPER.NAME))
    return DATASET_WRAPPER_REGISTRY.get(cfg.DATASET_WRAPPER.NAME)