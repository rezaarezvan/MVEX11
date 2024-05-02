import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

img = Image.open('dawg.png')
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)

noise = torch.randn_like(img) * 1.5
noisy_img = img + noise
noisy_img = torch.clamp(noisy_img, 0, 1)

plt.figure(figsize=(15, 9))
plt.subplot(1, 2, 1)
plt.imshow(img[0].permute(1, 2, 0))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_img[0].permute(1, 2, 0))
plt.axis('off')

plt.tight_layout()
plt.savefig('dawg_vs_noisy_dawg.png', bbox_inches='tight')
