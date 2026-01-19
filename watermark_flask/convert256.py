import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_binary_mask(input_path, output_path, size=(256, 256)):
    """
    读取二值图，将其缩放到 256x256 并保持二值化（0 和 1）。

    Args:
        input_path (str): 输入二值图路径
        output_path (str): 保存的二值图路径
        size (tuple): 目标尺寸 (默认 256x256)

    Returns:
        np.ndarray: 处理后的二值 mask
    """
    # 读取二值图 (确保是灰度图)
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 确保 mask 是二值化 (0 和 255)
    mask[mask > 0] = 255

    # 进行最近邻插值缩放，防止灰度值出现
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # 再次确保缩放后仍然是二值图（0 和 255）
    resized_mask[resized_mask > 0] = 255

    # 保存结果
    cv2.imwrite(output_path, resized_mask)

    # 显示结果
    plt.figure(figsize=(5, 5))
    plt.imshow(resized_mask, cmap="gray")
    plt.axis("off")
    plt.show()

    print(f"Resized binary mask saved as: {output_path}")

    return resized_mask


# 示例使用
input_mask_path = "/home/wh/ywy/watermark-anything-main/assets/masks/ducks_1.jpg"  # 你的二值图路径
output_mask_path = "/home/wh/ywy/watermark-anything-main/assets/masks/ducks_1_256.jpg"

resized_mask = resize_binary_mask(input_mask_path, output_mask_path)
