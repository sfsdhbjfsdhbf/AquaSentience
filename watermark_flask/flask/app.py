import os
import uuid
import logging
import sys

# =========================================================
# 重要说明：
# 由于本项目根目录下存在名为 "flask" 的目录（本文件所在目录），
# 如果用 `import flask` 会与 PyPI 的 Flask 包同名冲突。
# 因此这里通过 importlib 显式导入第三方 Flask 相关对象。
# =========================================================
import importlib

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WAM_DIR = os.path.join(_ROOT_DIR, "watermark-anything-main")

# 确保 watermark-anything-main 可以被导入
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

def _import_external_pkg(pkg_name: str):
    """
    由于项目存在本地目录 `flask/`，会遮蔽第三方 Flask 包。
    这里通过临时移除本地路径，确保导入的是 site-packages 里的包。
    """
    local_dirs = {
        os.path.abspath(os.path.dirname(__file__)),  # .../flask
        _ROOT_DIR,  # 项目根目录
        "",  # 有些情况下 CWD 以空串形式出现
    }

    removed = []
    for p in list(sys.path):
        ap = os.path.abspath(p) if p else ""
        if ap in local_dirs or p in local_dirs:
            try:
                sys.path.remove(p)
                removed.append(p)
            except ValueError:
                pass
    try:
        return importlib.import_module(pkg_name)
    finally:
        # 恢复原 sys.path 顺序（尽量保持一致）
        for p in reversed(removed):
            sys.path.insert(0, p)


flask_pkg = _import_external_pkg("flask")  # 第三方 Flask 包
Flask = flask_pkg.Flask
request = flask_pkg.request
jsonify = flask_pkg.jsonify

flask_cors_pkg = _import_external_pkg("flask_cors")
CORS = flask_cors_pkg.CORS

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint,
    default_transform,
    unnormalize_img,
)

# =========================================================
# 基础配置
# =========================================================
device = torch.device("cpu")
wam = None

EXP_DIR = os.path.join(_WAM_DIR, "checkpoints")
JSON_PATH = os.path.join(EXP_DIR, "params.json")
CKPT_PATH = os.path.join(EXP_DIR, "wam_mit.pth")

# 输出目录：从 outputs 改为项目根目录 static
STATIC_DIR = os.path.join(_ROOT_DIR, "static")
ORI_DIR = os.path.join(STATIC_DIR, "ori")
WAM_OUT_DIR = os.path.join(STATIC_DIR, "wam")
MASK_ORI_DIR = os.path.join(STATIC_DIR, "mask_ori")
MASK_PRE_DIR = os.path.join(STATIC_DIR, "mask_pre")

for d in [STATIC_DIR, ORI_DIR, WAM_OUT_DIR, MASK_ORI_DIR, MASK_PRE_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# 日志模块
# =========================================================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================
# 模型加载
# =========================================================
def load_model_once():
    global wam
    logger.info("===> 加载模型中...")
    # inference_utils 会从 params.json 里读取诸如 "configs/embedder.yaml" 之类的相对路径，
    # 这些路径是相对 watermark-anything-main/ 的。我们从项目根启动时会解析到错误位置，
    # 因此这里临时切到 watermark-anything-main/ 再加载，加载完恢复 cwd。
    old_cwd = os.getcwd()
    try:
        os.chdir(_WAM_DIR)
        wam = load_model_from_checkpoint(JSON_PATH, CKPT_PATH).to(device).eval()
    finally:
        os.chdir(old_cwd)
    logger.info("===> 模型加载完成")


# =========================================================
# Flask 应用
# - static_url_path: /static
# - static_folder:   项目根目录 static/
# =========================================================
app = Flask(__name__, static_url_path="/static", static_folder=STATIC_DIR)
CORS(app)  # 全局跨域支持


# =========================================================
# 工具函数
# =========================================================
def load_img_from_file(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img_pt = default_transform(img).unsqueeze(0).to(device)
    return img, img_pt


def load_mask_from_file(file_storage, target_hw):
    h, w = target_hw
    mask_img = Image.open(file_storage.stream).convert("L")
    mask_img = mask_img.resize((w, h), Image.NEAREST)
    mask_array = (np.array(mask_img) > 128).astype(np.float32)
    mask_tensor = torch.tensor(mask_array).unsqueeze(0).unsqueeze(0).to(device)
    return mask_img, mask_tensor


def text4_to_bits32(text: str) -> str:
    text = (text or "")[:4].ljust(4)
    return "".join(f"{ord(c):08b}" for c in text)


def bits32_to_text(bits: str) -> str:
    bits = bits[:32].ljust(32, "0")
    return "".join(chr(int(bits[i:i + 8], 2)) for i in range(0, 32, 8))


def bits_str_to_tensor(bits: str) -> torch.Tensor:
    bits = bits[:32].ljust(32, "0")
    # base_msg 仅用于 dtype/device 对齐
    base_msg = wam.get_random_msg(1).to(device)
    bit_list = [int(b) for b in bits]
    return torch.tensor(bit_list, dtype=base_msg.dtype, device=device).unsqueeze(0)


def msg_tensor_to_bits_str(msg_tensor: torch.Tensor) -> str:
    flat = msg_tensor.view(-1)[:32]
    return "".join(str(int(v.item())) for v in flat)


# =========================================================
# encode_full
# =========================================================
@app.route("/encode_full", methods=["POST"])
def encode_full():
    logger.info("收到 encode_full 请求")

    if "image" not in request.files:
        return jsonify({"error": "缺少 image 文件"}), 400

    image_file = request.files["image"]
    msg_text = request.form.get("msg", "")

    if not msg_text:
        return jsonify({"error": "缺少 msg 字段"}), 400

    orig_bits = text4_to_bits32(msg_text)
    wm_msg = bits_str_to_tensor(orig_bits)

    img_pil, img_pt = load_img_from_file(image_file)
    _, _, h, w = img_pt.shape

    ori_name = f"ori_{uuid.uuid4().hex}.png"
    img_pil.save(os.path.join(ORI_DIR, ori_name))
    logger.info(f"原图已保存: {ORI_DIR}/{ori_name}")

    mask_tensor = torch.ones((1, 1, h, w), dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = wam.embed(img_pt, wm_msg)
        imgs_w = outputs["imgs_w"]
        img_w = imgs_w * mask_tensor + img_pt * (1 - mask_tensor)

    wm_filename = f"wm_full_{uuid.uuid4().hex}.png"
    save_image(unnormalize_img(img_w), os.path.join(WAM_OUT_DIR, wm_filename))
    logger.info(f"水印图已保存: {WAM_OUT_DIR}/{wm_filename}")

    return jsonify({
        "msg_text": msg_text[:4],
        "bits_32": orig_bits,
        "watermarked_image_url": f"/static/wam/{wm_filename}",
        "original_image_url": f"/static/ori/{ori_name}",
    })


# =========================================================
# encode
# =========================================================
@app.route("/encode", methods=["POST"])
def encode():
    logger.info("收到 encode 请求")

    if "image" not in request.files:
        return jsonify({"error": "缺少 image 文件"}), 400
    if "mask" not in request.files:
        return jsonify({"error": "缺少 mask 文件"}), 400

    img_file = request.files["image"]
    mask_file = request.files["mask"]
    msg_text = request.form.get("msg", "")

    orig_bits = text4_to_bits32(msg_text)
    wm_msg = bits_str_to_tensor(orig_bits)

    img_pil, img_pt = load_img_from_file(img_file)
    ori_name = f"ori_{uuid.uuid4().hex}.png"
    img_pil.save(os.path.join(ORI_DIR, ori_name))
    logger.info(f"原图已保存: {ORI_DIR}/{ori_name}")

    _, _, h, w = img_pt.shape
    mask_pil, mask_tensor = load_mask_from_file(mask_file, (h, w))

    mask_ori_name = f"mask_ori_{uuid.uuid4().hex}.png"
    mask_pil.save(os.path.join(MASK_ORI_DIR, mask_ori_name))
    logger.info(f"原始 mask 已保存: {MASK_ORI_DIR}/{mask_ori_name}")

    with torch.no_grad():
        outputs = wam.embed(img_pt, wm_msg)
        img_w = outputs["imgs_w"] * mask_tensor + img_pt * (1 - mask_tensor)

    wm_name = f"wm_{uuid.uuid4().hex}.png"
    save_image(unnormalize_img(img_w), os.path.join(WAM_OUT_DIR, wm_name))
    logger.info(f"水印图已保存: {WAM_OUT_DIR}/{wm_name}")

    return jsonify({
        "msg_text": msg_text,
        "bits_32": orig_bits,
        "watermarked_image_url": f"/static/wam/{wm_name}",
        "mask_ori_url": f"/static/mask_ori/{mask_ori_name}",
        "original_image_url": f"/static/ori/{ori_name}",
    })


# =========================================================
# decode
# =========================================================
@app.route("/decode", methods=["POST"])
def decode():
    logger.info("收到 decode 请求")

    if "image" not in request.files:
        return jsonify({"error": "缺少 image 文件"}), 400

    img_file = request.files["image"]
    img_pil, img_pt = load_img_from_file(img_file)

    ori_name = f"ori_decode_{uuid.uuid4().hex}.png"
    img_pil.save(os.path.join(ORI_DIR, ori_name))

    _, _, h, w = img_pt.shape

    with torch.no_grad():
        preds = wam.detect(img_pt)["preds"]
        mask_preds = torch.sigmoid(preds[:, 0])
        bit_preds = preds[:, 1:, :, :]
        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()

    pred_bits_str = msg_tensor_to_bits_str(pred_message[0])
    decoded_text = bits32_to_text(pred_bits_str)

    mask_resized = F.interpolate(
        mask_preds.unsqueeze(1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    mask_pre_name = f"mask_pre_{uuid.uuid4().hex}.png"
    save_image(mask_resized, os.path.join(MASK_PRE_DIR, mask_pre_name))
    logger.info(f"预测 mask 已保存: {MASK_PRE_DIR}/{mask_pre_name}")

    return jsonify({
        "decoded_bits_32": pred_bits_str,
        "decoded_msg_text": decoded_text,
        "pred_mask_url": f"/static/mask_pre/{mask_pre_name}",
        "original_image_url": f"/static/ori/{ori_name}",
    })


# =========================================================
# 启动 Flask
# =========================================================
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        load_model_once()

    logger.info("服务启动中 http://0.0.0.0:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)


