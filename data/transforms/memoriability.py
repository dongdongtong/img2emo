from .build import TRANSFORM_REGISTRY


import torchvision.transforms as transforms

# all interpolations are BICUBIC
# all normalizations are CLIP's


@TRANSFORM_REGISTRY.register()
def clip_trivial_transform(cfg, is_train=True, only_other_transforms=False):
    
    img_size = cfg.INPUT.SIZE
    
    base_transform = [
        transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
        # NOTE!!!: This is the normalization used in CLIP preprocess
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ]
    
    if not is_train:
        return transforms.Compose(base_transform)
    
    if only_other_transforms:
        complex_transforms = [
            transforms.Resize(max(img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            
            transforms.RandomHorizontalFlip(),
            
            transforms.RandomGrayscale(0.01),
            
            transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BICUBIC),
            
            transforms.RandomPerspective(
                distortion_scale=0.2, p=0.5
            ),  # can be helpful if your images might have varying perspectives.
            
            transforms.RandomResizedCrop(img_size, scale=(0.08, 2.0)),
            
            transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
            # # NOTE!!!: This is the normalization used in CLIP preprocess
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
            ),  # Should help overfitting
        ]
    
    return transforms.Compose(complex_transforms)



@TRANSFORM_REGISTRY.register()
def resmem_transforms(cfg, is_train=True, only_other_transforms=False):
    
    transformer = transforms.Compose((
        transforms.Resize((256, 256)),
        transforms.CenterCrop(227),
        transforms.ToTensor()
        )
    )

    return transformer