import os

# CUDA memory allocator config to eliminate fragmentation.
# max_split_size_mb:128   — don't cache blocks >128MB (reduces large fragmentation)
# roundup_power2_divs:16  — round all sizes to 16 power-of-2 buckets so
#                           identical blocks are reused instead of fragmenting
# expandable_segments     — Linux-only, further defrag for PyTorch ≥ 2.0
import platform
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
for key in ["max_split_size_mb", "roundup_power2_divisions"]:
    if key not in alloc_conf:
        val = "128" if key == "max_split_size_mb" else "16"
        alloc_conf = key + ":" + val + ("," + alloc_conf if alloc_conf else "")
if platform.system() != "Windows" and "expandable_segments" not in alloc_conf:
    alloc_conf = "expandable_segments:True," + alloc_conf
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

from douzero.dmc import parser
from douzero.dmc.dmc import train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
