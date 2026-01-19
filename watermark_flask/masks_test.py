

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint, default_transform, unnormalize_img,
    create_random_mask, plot_outputs, msg2str, multiwm_dbscan,create_weird_shape_mask
)
import  cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_img(path):
    img = Image.open(path).convert("RGB")
    img = default_transform(img).unsqueeze(0).to(device)
    return img
# Load the model from the specified checkpoint
exp_dir = "checkpoints"
json_path = os.path.join(exp_dir, "params.json")
ckpt_path = os.path.join(exp_dir, 'wam_mit.pth')
wam = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

# Define the directory containing the images to watermark
img_dir = "assets/images"  # Directory containing the original images
output_dir = "outputs"  # Directory to save the watermarked images
os.makedirs(output_dir, exist_ok=True)

# DBSCAN parameters for detection
epsilon = 1  # min distance between decoded messages in a cluster
min_samples = 500  # min number of pixels in a 256x256 image to form a cluster

# multiple 32 bit message to hide (could be more than 2; does not have to be 1 minus the other)
wm_msgs = torch.randint(0, 2, (2, 32)).float().to(device)
proportion_masked = 0.1  # max proportion per watermark, randomly placed

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256 for consistency
    transforms.ToTensor(),
])





from notebooks.inference_utils import multiwm_dbscan

# DBSCAN parameters for detection
epsilon = 1 # min distance between decoded messages in a cluster
min_samples = 500 # min number of pixels in a 256x256 image to form a cluster

# multiple 32 bit message to hide (could be more than 2; does not have to be 1 minus the other)
wm_msgs = wam.get_random_msg(2)
print("Original messages: ", [msg2str(msg) for msg in wm_msgs])

# load image
img_pt = load_img(os.path.join(img_dir, "ducks.jpg"))

# load background
img_pt_background = load_img(os.path.join(img_dir, "seabackground.jpg"))
img_pt_background = F.interpolate(img_pt_background, size=(img_pt.shape[-2], img_pt.shape[-1]), mode="bilinear", align_corners=False)

multi_wm_img = img_pt_background.clone()

# Creates mask to use. `1` where the watermark will be placed, `0` elsewhere.
masks = []
for duck_nb in [1, 2]:
    mask_path = f'assets/masks/ducks_{duck_nb}.jpg'
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    target_shape = (img_pt.shape[-1], img_pt.shape[-2])
    mask = mask.resize(target_shape, Image.NEAREST)
    mask_array = np.array(mask)
    mask_array = (mask_array > 128).astype(np.float32)
    masks.append(mask_array)
masks = torch.tensor(np.array(masks)).to(device)

for ii in range(len(wm_msgs)):
    wm_msg, mask = wm_msgs[ii].unsqueeze(0), masks[ii]
    outputs = wam.embed(img_pt, wm_msg)
    multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)  # [1, 3, H, W]

# Detect the watermark in the multi-watermarked image
preds = wam.detect(multi_wm_img)["preds"]  # [1, 33, 256, 256]
mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
mask_preds_res = F.interpolate(mask_preds.unsqueeze(1), size=(img_pt.shape[-2], img_pt.shape[-1]), mode="nearest")  # [1, 1, H, W]
bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

centroids, positions = multiwm_dbscan(bit_preds, mask_preds, epsilon = epsilon, min_samples = min_samples)
centroids_pt = torch.stack(list(centroids.values()))
plot_outputs(img_pt.detach(), multi_wm_img.detach(), masks.sum(0).detach(), (mask_preds_res>0.5).float().detach(), labels = positions, centroids = centroids)
# positions has the cluster number at each pixel. can be upsaled back to the original size.

print(f"number messages found in image {img_pt}: {len(centroids)}")
for centroid in centroids_pt:
    print(f"found centroid: {msg2str(centroid)}")
    bit_acc = (centroid == wm_msgs).float().mean(dim=1)
    # get message with maximum bit accuracy
    bit_acc, idx = bit_acc.max(dim=0)
    hamming = int(torch.sum(centroid != wm_msgs[idx]).item())
    print(f"bit accuracy: {bit_acc.item()} - hamming distance: {hamming}/{len(wm_msgs[0])}")