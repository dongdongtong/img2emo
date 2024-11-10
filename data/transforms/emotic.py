from .build import TRANSFORM_REGISTRY


import torchvision.transforms as transforms


@TRANSFORM_REGISTRY.register()
def emotic_transform(cfg, is_train=True, only_other_transforms=False):
    
    img_size = cfg.INPUT.SIZE
    
    base_transform = [
        transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    if not is_train:
        return transforms.Compose(base_transform)
    
    if only_other_transforms:
        complex_transforms = [
            transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomGrayscale(0.01),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # model more robust to changes in lighting conditions.
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # can be helpful if your images might have varying perspectives.
            transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),  # Should help overfitting
        ]
    
    return transforms.Compose(complex_transforms)


@TRANSFORM_REGISTRY.register()
def trivial_transform(cfg, is_train=True, only_other_transforms=False):
    
    img_size = cfg.INPUT.SIZE
    
    base_transform = [
        transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    if not is_train:
        return transforms.Compose(base_transform)
    
    if only_other_transforms:
        complex_transforms = [
            transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            
            transforms.RandomHorizontalFlip(),
            
            transforms.RandomGrayscale(0.01),
            
            transforms.TrivialAugmentWide(),
            
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # can be helpful if your images might have varying perspectives.
            
            transforms.RandomResizedCrop(img_size, scale=(0.08, 2.0)),
            
            transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),  # Should help overfitting
        ]
    
    return transforms.Compose(complex_transforms)


@TRANSFORM_REGISTRY.register()
def weak_trivial_transform(cfg, is_train=True, only_other_transforms=False):
    
    img_size = cfg.INPUT.SIZE
    
    base_transform = [
        transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    if not is_train:
        return transforms.Compose(base_transform)
    
    if only_other_transforms:
        complex_transforms = [
            transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            
            transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    
    return transforms.Compose(complex_transforms)


@TRANSFORM_REGISTRY.register()
def strong_trivial_transform(cfg, is_train=True, only_other_transforms=False):
    
    img_size = cfg.INPUT.SIZE
    
    base_transform = [
        transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    
    if not is_train:
        return transforms.Compose(base_transform)
    
    if only_other_transforms:
        complex_transforms = [
            transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            
            transforms.TrivialAugmentWide(),
            
            transforms.RandomResizedCrop(img_size, scale=(0.08, 2.0)),
            
            transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),  # Should help overfitting
        ]
    
    return transforms.Compose(complex_transforms)