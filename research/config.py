"""Shared constants for the offline experiments.

The preprocessing values below are the single source of truth: the Swift app must
copy them exactly (resize, crop, interpolation, mean, std, channel order, pixel scale).
"""

# --- Preprocessing (applied identically to every backbone) -------------------
RESIZE_SHORT_SIDE = 256          # resize so the shorter side is 256, keep aspect ratio
CROP_SIZE = 224                  # then center crop 224 x 224
INTERPOLATION = "bilinear"       # torchvision Resize default (with antialiasing for PIL images)
PIXEL_SCALE = 1.0 / 255.0        # uint8 [0, 255] -> float [0, 1] before normalizing
CHANNEL_ORDER = "RGB"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Backbones ---------------------------------------------------------------
# MobileNetV3-Large: torchvision IMAGENET1K_V1 weights. Chosen because its own recipe
# uses resize 256 / crop 224 (V2 weights use resize 232, which would not match above).
MOBILENET_WEIGHTS = "IMAGENET1K_V1"
# DINOv2-small (ViT-S/14, distilled), loaded via torch.hub from facebookresearch/dinov2.
# 224 is divisible by the patch size 14, so no padding is needed.
DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_HUB_MODEL = "dinov2_vits14"

BACKBONES = ("mobilenetv3", "dinov2")      # names used in output filenames
DATASETS = ("flowers102", "pets")          # names used in output filenames

# --- Runtime -----------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 0                                   # for the --limit subset selection
DATA_DIR = "research/data"                 # dataset downloads (relative to repo root)
RESULTS_DIR = "results"                    # output .npz files (relative to repo root)

# --- Experiments 5 and 6 -----------------------------------------------------
N_SEEDS = 100             # random trials per (dataset, backbone, head) cell
K_SHOT = 5                # support examples per taught class
N_QUERY = 15              # held-out query examples per class
EXP5_N_CLASSES = 10       # classes added one at a time
EXP6_N_TAUGHT = 10        # taught classes
EXP6_N_UNKNOWN = 10       # held-out "unknown" classes
CI_LEVEL = 0.95

# --- Heads: must match PersonalizeCore exactly -------------------------------
LOGISTIC_STEPS = 300      # LogisticHead.init(steps:)
LOGISTIC_LR = 1.0         # LogisticHead.init(learningRate:)
LOGISTIC_REG = 1e-4       # LogisticHead.init(regularization:)
NCM_TEMPERATURE = 0.1     # NearestClassMeanHead(temperature:)
THRESHOLD_STD_DEVS = 2.0  # NearestClassMeanHead.suggestedThreshold(stdDevs:)
ECE_BINS = 15
