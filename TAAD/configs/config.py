"""
Central configuration for the TAAD + GNN Action Detection Module.
Edit values here; every other file imports from this module.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Video / clip settings  (Section 3.2.3)
# ──────────────────────────────────────────────────────────────────────────────
CLIP_FRAMES      = 50          # T — number of frames per clip (paper: 50)
FRAME_RATE       = 25          # FPS used during sampling (paper: 25 FPS)
IMG_H            = 352         # spatial height after resize  (paper: 640×352)
IMG_W            = 640         # spatial width  after resize

# ──────────────────────────────────────────────────────────────────────────────
# X3D backbone  (Section 3.2.2)
# ──────────────────────────────────────────────────────────────────────────────
X3D_MODEL        = "x3d_s"     # options: x3d_xs | x3d_s | x3d_m | x3d_l
#                               paper uses X3D-S for single-GPU budget
X3D_FEAT_DIM     = 192         # D  — output channel dim of X3D block-5
FPN_OUT_DIM      = 192         # same as X3D_FEAT_DIM (FPN keeps channels)

# ──────────────────────────────────────────────────────────────────────────────
# ROI Align  (Section 3.2.2)
# ──────────────────────────────────────────────────────────────────────────────
ROI_OUTPUT_SIZE  = (4, 4)      # spatial pooling output  (H×W after ROI Align)
ROI_SPATIAL_SCALE = 1.0 / 32   # stride from backbone to feature map

# ──────────────────────────────────────────────────────────────────────────────
# GNN — Local Game State  (Section 3.3)
# ──────────────────────────────────────────────────────────────────────────────
MAX_PLAYERS      = 22          # N — max players per frame (both teams)
K_NEIGHBORS      = 6           # M closest players to connect per node (paper: 6)
GNN_LAYERS       = 3           # number of EdgeConv layers  (paper: 3 or 4)

#  Node feature vector x_i_t  (Equation 2):
#    pos_x, pos_y       2
#    vel_x, vel_y       2
#    team_membership    1
#    visual_proj        VISUAL_PROJ_DIM
VISUAL_PROJ_DIM  = 64          # D' — dimension of projected visual feature
NODE_FEAT_DIM    = 2 + 2 + 1 + VISUAL_PROJ_DIM   # = 69

GNN_HIDDEN_DIM   = 128         # hidden width inside EdgeConv MLP
GNN_OUT_DIM      = 128         # output embedding dimension per node

# ──────────────────────────────────────────────────────────────────────────────
# Temporal Convolutional Network — TCN  (Section 3.2.1 / TAAD)
# ──────────────────────────────────────────────────────────────────────────────
TCN_CHANNELS     = [256, 256, 256]   # channels in each TCN layer
TCN_KERNEL       = 3                 # temporal kernel size
TCN_DROPOUT      = 0.3

# Combined feature dim fed into TCN:  X3D_FEAT_DIM + GNN_OUT_DIM
COMBINED_DIM     = X3D_FEAT_DIM + GNN_OUT_DIM    # 192 + 128 = 320

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE       = 6           # paper: batch size 6 on RTX A6000
LR               = 5e-4        # paper: 0.0005
WEIGHT_DECAY     = 1e-5        # paper: 1e-5 on non-bias params
GRAD_ACCUM_STEPS = 20          # paper: gradient accumulation over 20 iterations
NUM_EPOCHS       = 13          # paper: 13 epochs
LR_DROP_EPOCH    = 10          # divide LR by 10 at this epoch

# ──────────────────────────────────────────────────────────────────────────────
# Inference / action-tube construction  (Section 3.5)
# ──────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.5         # confidence score threshold (paper high-recall mode)
IOU_THRESHOLD    = 0.2         # temporal IoU threshold for tube matching
SMOOTH_WINDOW    = 5           # label-smoothing window for tube construction

# ──────────────────────────────────────────────────────────────────────────────
# Game-state proxy (until homography is available)
# ──────────────────────────────────────────────────────────────────────────────
# When real pitch coordinates are unavailable we use normalised screen-space
# bbox centres as a stand-in position.  Velocity is set to 0.
# Once homography + 2-D reconstruction is wired in, replace
# `utils/game_state.py::extract_game_state()` with real pitch coords.
USE_PROXY_POSITIONS = True     # set False once homography is ready
