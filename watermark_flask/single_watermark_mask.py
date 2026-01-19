"""
兼容入口（旧路径）：
原来服务在 `watermark-anything-main/single_watermark_mask.py`。
现在已迁移到项目根目录 `flask/app.py`，并将输出目录从 outputs 改为 static。

保留该文件是为了不破坏旧的启动方式：
`python watermark-anything-main/single_watermark_mask.py`
"""

import os
import runpy


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ""))
    target = os.path.join(root, "flask", "app.py")
    runpy.run_path(target, run_name="__main__")
