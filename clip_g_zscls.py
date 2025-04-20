# import torch
# from PIL import Image
# import open_clip
# from utils.scenes import template, templates, scenes

# import torchvision.transforms as transforms

# base_transform = transforms.Compose([
#     transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
# print(preprocess)
# model = model.to("cuda")
# # model.visual = model.visual.to("cuda")
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer('ViT-g-14')

# image = base_transform(Image.open("0001.jpg")).unsqueeze(0)

# texts = tokenizer([template + scene for scene in scenes])

# texts = []
# for template in templates:
#     texts.append(tokenizer(template + "a photo of a " + scene for scene in scenes))

# # text = tokenizer(["an image of mountain", "a dog", "a cat"])

# with torch.no_grad(), torch.autocast("cuda"):
    
#     image = image.to("cuda")
#     texts = texts.to("cuda")
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(texts)  # Uncommented this line to encode text features
#     # save text features
#     # torch.save(text_features, "/data/dingsd/img2emo/image_memoriability/datasets/scenes401_PlaceDataset.pt")
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     print("Image features shape:", image_features.shape)
#     print("Text features shape:", text_features.shape)
#     print("Probability shape:", text_probs.shape)
    
#     top_probs, indices = text_probs[0].topk(5)
#     print("Top 5 most similar scenes:")
#     for i in range(5):
#         print(f"{i+1}: {scenes[indices[i]]} - {top_probs[i].item():.4f}")

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

# # load text features
# text_features = torch.load("/data/dingsd/img2emo/image_memoriability/datasets/scenes401_PlaceDataset.pt", weights_only=False, map_location="cpu")
# print("Text features shape:", text_features.shape)



from torchvision.models import maxvit_t

a = maxvit_t()
params = sum(p.numel() for p in a.parameters())
print(f"Total parameters in maxvit_t: {(params / 1e6):.2f}M")

from timm.models.vision_transformer import vit_giant_patch14_clip_224

b = vit_giant_patch14_clip_224(pretrained=False)
params = sum(p.numel() for p in b.parameters())
print(f"Total parameters in vit_giant_patch14_clip_224: {(params / 1e6):.2f}M")
# Total parameters in maxvit_t: 30.92M
# Total parameters in vit_giant_patch14_clip_224: 1012.65M