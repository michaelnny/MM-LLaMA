import torch


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.models.imagebind import data
from mm_llama.models.imagebind.models.imagebind_model import imagebind_huge, ModalityType


ckpt_path = '/home/michael/models/ImageBind/imagebind_huge.pth'

image_paths = ['.assets/dog_image.jpg', '.assets/car_image.jpg']
video_paths = ['.assets/dog_video.mp4', '.assets/car_video.mp4']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Instantiate model
model = imagebind_huge(ckpt_path)
model.eval()
model.to(device)

# For videos
inputs = {
    ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

# [2, 15, 3, 2, 224, 224]
print(inputs[ModalityType.VISION].shape)

# [2, 1024]
print(embeddings[ModalityType.VISION].shape)


# For images
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

# [2, 3, 224, 224]
print(inputs[ModalityType.VISION].shape)

# [2, 1024]
print(embeddings[ModalityType.VISION].shape)
