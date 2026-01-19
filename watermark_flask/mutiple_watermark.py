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

# Iterate through images in the image directory
for img_ in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Mask to use. 1 values correspond to pixels where the watermark will be placed.
    height, width = 256, 256
    mask_filename = "/home/wh/ywy/watermark-anything-main/assets/masks/ducks_1_256.jpg"
    mask_256 = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    masks = (mask_256 / 255).astype(np.uint8)
    masks = torch.tensor(masks, dtype=torch.float32).to(img_tensor.device)
    # masks = create_weird_shape_mask(img_tensor,height, width )
    # masks = create_random_mask(img_tensor, num_masks=len(wm_msgs), mask_percentage=proportion_masked)  # create one random mask per message
    multi_wm_img = img_tensor.clone()

    # Embed watermarks into the image
    for ii in range(len(wm_msgs)):
        wm_msg, mask = wm_msgs[ii].unsqueeze(0), masks[ii]
        outputs = wam.embed(img_tensor, wm_msg)
        multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)  # [1, 3, H, W]

    # Detect the watermark in the multi-watermarked image
    preds = wam.detect(multi_wm_img)["preds"]  # [1, 33, 256, 256]
    mask_preds = F.sigmoid(preds[:, 0, :, :])  # [1, 256, 256], predicted mask
    bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits

    # positions has the cluster number at each pixel. can be upsaled back to the original size.
    centroids, positions = multiwm_dbscan(bit_preds, mask_preds, epsilon=epsilon, min_samples=min_samples)
    centroids_pt = torch.stack(list(centroids.values())).to(device)

    print(f"Number of messages found in image {img_}: {len(centroids)}")
    for centroid in centroids_pt:
        print(f"Found centroid: {msg2str(centroid)}")
        bit_acc = (centroid == wm_msgs).float().mean(dim=1)
        # Get message with maximum bit accuracy
        bit_acc, idx = bit_acc.max(dim=0)
        hamming = int(torch.sum(centroid != wm_msgs[idx]).item())
        print(f"Bit accuracy: {bit_acc.item()} - Hamming distance: {hamming}/{len(wm_msgs[0])}")

    # Convert mask predictions to binary
    binary_mask = (mask_preds.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Save the binary mask image
    mask_img = Image.fromarray(binary_mask)
    mask_img.save(os.path.join(output_dir, f"{os.path.splitext(img_)[0]}_mask.png"))

    print(f"Saved binary mask for {img_} to {output_dir}")
