import os, cv2, gc, time, math, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader, random_split,
                              WeightedRandomSampler, ConcatDataset)
from torch.optim.swa_utils import AveragedModel, update_bn
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, average_precision_score, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score,
    brier_score_loss, log_loss, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

FF_FRAME_ROOT = "/kaggle/input/datasets/syedazmulhasansabbir/face-forensics-frames/Face Forensics Frames"
CDF_FRAME_ROOT = "/kaggle/input/datasets/syedazmulhasansabbir/celeb-df-frames/content/drive/MyDrive/dataset/celeb_df_frames"
FF_REAL_FOLDER = 'orginal'
FF_FAKE_FOLDERS = ['Face2Face', 'Deepfakes', 'NeuralTextures', 'FaceShifter', 'FaceSwap']

TEACHER_A_PATH = "/kaggle/input/models/syedazmulhasansabbir/kd-ff-teach-dp-std-efcnt/tensorflow2/default/1/KD_student_EfficientNet_teacher_DP_ViT_FF.pth"
TEACHER_B_PATH = "/kaggle/input/models/syedazmulhasansabbir/kd-celeb-df-teacher-dp-vit-std-efficientnet/tensorflow2/default/1/kd_p2r_student_epoch25_BEST.pth"

# ── BACKBONE ─────────────────────────────────────────────────────────────────
BACKBONE_NAME = 'efficientnet_b2'  # kept from V14

# ── PATHS ────────────────────────────────────────────────────────────────────
RESUME_CHECKPOINT = None
OUT_DIR = Path("/kaggle/working/dual_teacher_kd_v15_outputs")
FIG_DIR = OUT_DIR / "figures"
CKPT_DIR = OUT_DIR / "checkpoints"
for d in [OUT_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42

# V15 FIX-10: 30 epochs, enough clean training time
EPOCHS = 30
MAX_STEPS_PER_EPOCH = 1500
WARMUP_EPOCHS = 3
N_RESTARTS = 1  # Single cosine cycle, no restarts
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 10  # Patience only — no DROP_THRESHOLD
# V15 FIX-3: SWA starts at epoch 20 (after enough clean training)
SWA_START_EPOCH = 20

# ── LOSS WEIGHTS ──────────────────────────────────────────────────────────────
KD_TEMPERATURE = 1.5  # kept from V14 (good calibration)
# V15 FIX-6: Raise ALPHA_MIN/BETA_MIN from 0.05 → 0.15 to keep teacher signal
ALPHA_MAX = 0.30
ALPHA_MIN = 0.15  # V14 had 0.05 — KD decayed too aggressively
BETA_MAX = 0.30
BETA_MIN = 0.15  # same
# V15 FIX-5: ETA=1.0 to pair with cosine similarity loss (was 0.25 + L2)
ETA = 1.0
GAMMA = 1.0  # hard focal loss weight
LABEL_SMOOTHING = 0.0  # kept from V14 — teacher soft targets handle smoothing
LAMBDA_CONFLICT = 0.3
FOCAL_GAMMA = 0.5

# ── CROSS-DOMAIN (disabled — teachers near-random cross-domain) ───────────────
CROSS_DOMAIN_W = 0.0

# ── DANN (disabled — was erasing deepfake artifacts) ─────────────────────────
USE_DANN = False

# ── MIXUP — V15 FIX-1: DOMAIN-AWARE (only FF++, never CDF) ──────────────────
USE_MIXUP = True
MIXUP_ALPHA = 0.2
MIXUP_WARMUP_EPOCHS = 10  # No MixUp before epoch 10
MIXUP_RAMP_EPOCHS = 5  # Linear ramp-up alpha 0→0.2 over 5 epochs after warmup
# V15 FIX-2: Turn MixUp OFF for final 5 epochs → clean convergence
MIXUP_TURNOFF_EPOCHS = 5  # Last N epochs: no MixUp

# ── 6-CHANNEL INPUT (kept from V14) ──────────────────────────────────────────
USE_HF_CHANNEL = True

# ── TIME BUDGET ───────────────────────────────────────────────────────────────
MAX_RUN_TIME = 12.0 * 3600
EVAL_RESERVE = 120 * 60
EVAL_BUDGET_TEACHER = 50 * 60
EVAL_BUDGET_OPTIONAL = 35 * 60
EVAL_BUDGET_EXTRAS = 15 * 60

START_TIME = time.time()


def time_left(): return MAX_RUN_TIME - (time.time() - START_TIME)


def elapsed():   return time.time() - START_TIME


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_use_amp = DEVICE.type == 'cuda'

print(f"Device            : {DEVICE}")
print(f"MAX_RUN_TIME      : {MAX_RUN_TIME / 3600:.1f}h")
print(f"Backbone          : {BACKBONE_NAME}")
print(f"Input channels    : {'6 (RGB+HF)' if USE_HF_CHANNEL else '3 (RGB only)'}")
print(f"KD Temperature    : T={KD_TEMPERATURE}")
print(f"ETA (feat KD)     : {ETA} (cosine similarity loss, was 0.25+L2)")
print(f"ALPHA_MIN/MAX     : {ALPHA_MIN}/{ALPHA_MAX} (was 0.05/0.30 in V14)")
print(f"MixUp             : FF++ ONLY (CDF excluded — imbalanced 14%/86%)")
print(f"MixUp warmup      : ep {MIXUP_WARMUP_EPOCHS} + ramp {MIXUP_RAMP_EPOCHS} ep")
print(f"MixUp turnoff     : last {MIXUP_TURNOFF_EPOCHS} epochs (clean convergence)")
print(f"SWA start         : epoch {SWA_START_EPOCH} (after clean training)")
print(f"Calibration       : skipped if T ∈ [0.8, 1.2] (T=1.103 hurt ECE in V14)")

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATASETS (with HF channel + MULTITHREADED FAST SCAN)
# ══════════════════════════════════════════════════════════════════════════════
from concurrent.futures import ThreadPoolExecutor


def scan_single_dir(path):
    """Scans exactly one directory level quickly using os.scandir."""
    valid = {'.jpg', '.jpeg', '.png'}
    files, subdirs = [], []
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                if os.path.splitext(entry.name)[1].lower() in valid:
                    files.append(entry.path)
            elif entry.is_dir():
                subdirs.append(entry.path)
    except Exception:
        pass
    return files, subdirs


def parallel_fast_scan(root_dir):
    """Fires 32 parallel threads to blast through Kaggle's NFS bottleneck."""
    if not os.path.exists(root_dir):
        return []

    all_files = []
    dirs_to_scan = [root_dir]

    with ThreadPoolExecutor(max_workers=32) as executor:
        while dirs_to_scan:
            results = list(executor.map(scan_single_dir, dirs_to_scan))
            dirs_to_scan = []
            for f, d in results:
                all_files.extend(f)
                dirs_to_scan.extend(d)

    return all_files


class FFDataset(Dataset):
    def __init__(self, root_dir):
        self.real_paths, self.fake_paths = [], []
        self.path_to_manip = {}
        print("Scanning FF++ (using 32-thread parallel scan) …")

        real_dir = os.path.join(root_dir, FF_REAL_FOLDER)
        real_files = parallel_fast_scan(real_dir)
        self.real_paths.extend(real_files)
        for f in real_files: self.path_to_manip[f] = 'real'

        for name in FF_FAKE_FOLDERS:
            fake_dir = os.path.join(root_dir, name)
            fake_files = parallel_fast_scan(fake_dir)
            self.fake_paths.extend(fake_files)
            for f in fake_files: self.path_to_manip[f] = name

        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        self.idx_to_manip = [self.path_to_manip[p] for p in self.all_paths]
        print(f"  FF++  REAL={len(self.real_paths):,}  FAKE={len(self.fake_paths):,}")
        if not self.all_paths: raise ValueError("No FF++ images found.")

    def __len__(self):
        return len(self.all_paths)

    @staticmethod
    def _hf(img):
        a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return Image.fromarray(cv2.cvtColor(
            cv2.convertScaleAbs(cv2.Laplacian(a, cv2.CV_64F)), cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        img = Image.open(self.all_paths[idx]).convert('RGB')
        return img, self._hf(img), self.labels[idx], 0


class CDFDataset(Dataset):
    def __init__(self, root_dir):
        self.real_paths, self.fake_paths = [], []
        self.path_to_manip = {}
        print("Scanning Celeb-DF (using 32-thread parallel scan) …")

        for cls in ["real", "fake"]:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"  ⚠ Not found: {cls_dir}");
                continue

            files = parallel_fast_scan(cls_dir)
            if cls == "real":
                self.real_paths.extend(files)
            else:
                self.fake_paths.extend(files)

            for f in files: self.path_to_manip[f] = cls

        self.all_paths = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        self.idx_to_manip = [self.path_to_manip[p] for p in self.all_paths]
        print(f"  CDF   REAL={len(self.real_paths):,}  FAKE={len(self.fake_paths):,}")
        if not self.all_paths: raise ValueError("No CDF images found.")

    def __len__(self):
        return len(self.all_paths)

    @staticmethod
    def _hf(img):
        a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return Image.fromarray(cv2.cvtColor(
            cv2.convertScaleAbs(cv2.Laplacian(a, cv2.CV_64F)), cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        img = Image.open(self.all_paths[idx]).convert('RGB')
        return img, self._hf(img), self.labels[idx], 1


_TRAIN_RGB_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.15, 0.05),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_TRAIN_HF_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
_EVAL_RGB_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_EVAL_HF_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def make_6ch(rgb_t, hf_t):
    return torch.cat([rgb_t, hf_t], dim=0)


class TrainWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img, hf, lbl, dom = self.subset[idx]
        rgb_t = _TRAIN_RGB_TF(img)
        hf_t = _TRAIN_HF_TF(hf)
        inp = make_6ch(rgb_t, hf_t) if USE_HF_CHANNEL else rgb_t
        return (inp,
                torch.tensor(lbl, dtype=torch.float32),
                torch.tensor(dom, dtype=torch.long))


class EvalWrapper(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self): return len(self.subset)

    def __getitem__(self, idx):
        img, hf, lbl, dom = self.subset[idx]
        rgb_t = _EVAL_RGB_TF(img)
        hf_t = _EVAL_HF_TF(hf)
        inp = make_6ch(rgb_t, hf_t) if USE_HF_CHANNEL else rgb_t
        return (inp,
                torch.tensor(lbl, dtype=torch.float32),
                torch.tensor(dom, dtype=torch.long),
                torch.tensor(idx, dtype=torch.long))


torch.manual_seed(SEED)
ff_full = FFDataset(FF_FRAME_ROOT)
cdf_full = CDFDataset(CDF_FRAME_ROOT)


def split_ds(ds):
    n = len(ds);
    n_tr = int(0.8 * n);
    n_v = int(0.1 * n)
    g = torch.Generator().manual_seed(SEED)
    return random_split(ds, [n_tr, n_v, n - n_tr - n_v], generator=g)


ff_tr_sub, ff_v_sub, ff_te_sub = split_ds(ff_full)
cdf_tr_sub, cdf_v_sub, cdf_te_sub = split_ds(cdf_full)

print(f"\nFF++  tr={len(ff_tr_sub):,} val={len(ff_v_sub):,} te={len(ff_te_sub):,}")
print(f"CDF   tr={len(cdf_tr_sub):,} val={len(cdf_v_sub):,} te={len(cdf_te_sub):,}")

ff_lbl_tr = [ff_full.labels[i] for i in ff_tr_sub.indices]
cdf_lbl_tr = [cdf_full.labels[i] for i in cdf_tr_sub.indices]
ff_n0 = ff_lbl_tr.count(0);
ff_n1 = ff_lbl_tr.count(1)
cdf_n0 = cdf_lbl_tr.count(0);
cdf_n1 = cdf_lbl_tr.count(1)

# V15 FIX-9: Cap CDF pos_weight at 3.0 (was 5.0) — sampler already balances
FF_POS_WEIGHT = torch.tensor([min(ff_n0 / max(ff_n1, 1), 5.0)], device=DEVICE)
CDF_POS_WEIGHT = torch.tensor([min(cdf_n0 / max(cdf_n1, 1), 3.0)], device=DEVICE)  # ← 3.0 cap
print(f"FF++ pos_weight={FF_POS_WEIGHT.item():.3f}  CDF pos_weight={CDF_POS_WEIGHT.item():.3f}")

# Bias inits: conservative 0.5× to avoid over-biasing
FF_BIAS_INIT = math.log(max(ff_n1, 1) / max(ff_n0, 1))
CDF_BIAS_INIT = math.log(max(cdf_n1, 1) / max(cdf_n0, 1))
print(f"FF++ bias={FF_BIAS_INIT:.3f}  CDF bias={CDF_BIAS_INIT:.3f}")

# Weighted sampler: balance both domain and class
gc_ = {(0, 0): ff_n0, (0, 1): ff_n1, (1, 0): cdf_n0, (1, 1): cdf_n1}
sw = ([1 / max(gc_[(0, l)], 1) for l in ff_lbl_tr] +
      [1 / max(gc_[(1, l)], 1) for l in cdf_lbl_tr])
sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

_nw = 2
combined_train_ds = ConcatDataset([TrainWrapper(ff_tr_sub), TrainWrapper(cdf_tr_sub)])
train_loader = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=_nw, pin_memory=_use_amp,
                          prefetch_factor=2 if _nw > 0 else None,
                          persistent_workers=_nw > 0)

ff_val_loader = DataLoader(EvalWrapper(ff_v_sub), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=_nw, pin_memory=_use_amp)
cdf_val_loader = DataLoader(EvalWrapper(cdf_v_sub), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=_nw, pin_memory=_use_amp)
ff_test_loader = DataLoader(EvalWrapper(ff_te_sub), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=_nw, pin_memory=_use_amp)
cdf_test_loader = DataLoader(EvalWrapper(cdf_te_sub), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=_nw, pin_memory=_use_amp)

effective_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH or 999999)
print(f"\nCombined train: {len(combined_train_ds):,}  steps/epoch: {effective_steps:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TEACHERS
# ══════════════════════════════════════════════════════════════════════════════

class PeerTeacher(nn.Module):
    def __init__(self, feat_dim=1536):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        fd = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fd, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1))
        self.proj_head = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, feat_dim))

    def forward(self, x): return self.classifier(self.backbone(x))

    def get_features_and_logit(self, x):
        feat = self.backbone(x)
        logit = self.classifier(feat)
        return feat, logit


def load_teacher(path, name, feat_dim=1536):
    print(f"\nLoading {name} …")
    raw = torch.load(path, map_location=DEVICE, weights_only=False)
    sd = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    if 'proj_head.1.weight' in sd:
        feat_dim = sd['proj_head.1.weight'].shape[0]
    t = PeerTeacher(feat_dim).to(DEVICE)
    miss, unex = t.load_state_dict(sd, strict=False)
    rm = [k for k in miss if 'num_batches' not in k]
    ru = [k for k in unex if 'num_batches' not in k]
    print(f"  {'✅' if not rm and not ru else f'⚠ miss={len(rm)} unex={len(ru)}'}  "
          f"params={sum(p.numel() for p in t.parameters()) / 1e6:.1f}M")
    t.eval()
    for p in t.parameters(): p.requires_grad_(False)
    return t


teacher_A = load_teacher(TEACHER_A_PATH, "Teacher A (FF++)")
teacher_B = load_teacher(TEACHER_B_PATH, "Teacher B (CDF)")

with torch.no_grad():
    _d = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    fA_t, lA_t = teacher_A.get_features_and_logit(_d)
    fB_t, lB_t = teacher_B.get_features_and_logit(_d)
    TEACHER_FEAT_DIM = fA_t.shape[1]
    assert lA_t.shape == (2, 1)
print(f"  ✅ Teachers verified. Teacher feat dim: {TEACHER_FEAT_DIM}")
del _d, fA_t, lA_t, fB_t, lB_t;
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — STUDENT V15
# ══════════════════════════════════════════════════════════════════════════════

class DualStudent(nn.Module):
    def __init__(self, ff_bias_init: float = 0.0, cdf_bias_init: float = 0.0):
        super().__init__()

        # ── Backbone: EfficientNet-B2 ────────────────────────────────────────
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)
        d = self.backbone.num_features  # B2: 1408

        # Modify first conv 3→6 for RGB+HF input
        if USE_HF_CHANNEL:
            conv_attr = 'conv_stem' if hasattr(self.backbone, 'conv_stem') else 'first_conv'
            old_conv = getattr(self.backbone, conv_attr)
            out_ch = old_conv.out_channels
            k = old_conv.kernel_size;
            s = old_conv.stride;
            p = old_conv.padding
            bias_flag = old_conv.bias is not None
            new_conv = nn.Conv2d(6, out_ch, kernel_size=k, stride=s,
                                 padding=p, bias=bias_flag)
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                new_conv.weight[:, 3:, :, :] = (
                        old_conv.weight.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1) * 0.5)
                if bias_flag:
                    new_conv.bias.data = old_conv.bias.data.clone()
            setattr(self.backbone, conv_attr, new_conv)
            print(f"  [V15] Backbone first conv modified: 3ch→6ch (RGB+HF)")

        # ── Expert heads ─────────────────────────────────────────────────────
        self.expert_A = nn.Sequential(
            nn.Linear(d, d, bias=True), nn.GELU(), nn.Dropout(0.1))
        self.expert_B = nn.Sequential(
            nn.Linear(d, d, bias=True), nn.GELU(), nn.Dropout(0.1))

        # ── Soft domain gate ─────────────────────────────────────────────────
        self.domain_gate = nn.Embedding(2, d)
        nn.init.normal_(self.domain_gate.weight, mean=0.5, std=0.1)

        # ── Dual domain-specific classifiers ─────────────────────────────────
        self.classifier_ff = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1))
        nn.init.constant_(self.classifier_ff[-1].bias, ff_bias_init)

        self.classifier_cdf = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1))
        nn.init.constant_(self.classifier_cdf[-1].bias, cdf_bias_init)

        print(f"  [V15] FF++ classifier bias init: {ff_bias_init:.3f}")
        print(f"  [V15] CDF  classifier bias init: {cdf_bias_init:.3f}")

        # ── Separate projection heads per teacher (V14 FIX-4 — keep) ─────────
        self.proj_head_A = nn.Sequential(nn.Linear(d, TEACHER_FEAT_DIM), nn.GELU())
        self.proj_head_B = nn.Sequential(nn.Linear(d, TEACHER_FEAT_DIM), nn.GELU())

    def forward_full(self, x, domain_ids=None):
        feat = self.backbone(x)

        fA = self.expert_A(feat)
        fB = self.expert_B(feat)

        if domain_ids is not None:
            gate = torch.sigmoid(self.domain_gate(domain_ids))
        else:
            gate = torch.full_like(feat, 0.5)

        fused = gate * fA + (1.0 - gate) * fB

        # Route to domain-specific classifier
        if domain_ids is not None:
            logit_ff = self.classifier_ff(fused)
            logit_cdf = self.classifier_cdf(fused)
            is_ff = (domain_ids == 0).float().unsqueeze(1)
            logit = is_ff * logit_ff + (1.0 - is_ff) * logit_cdf
        else:
            logit = 0.5 * self.classifier_ff(fused) + 0.5 * self.classifier_cdf(fused)

        proj_A = self.proj_head_A(feat)
        proj_B = self.proj_head_B(feat)

        return logit, feat, fA, fB, fused, gate, (proj_A, proj_B)

    def forward(self, x, domain_ids=None):
        logit, _, _, _, _, _, _ = self.forward_full(x, domain_ids=domain_ids)
        return logit

    def get_features(self, x):
        return self.backbone(x)


student = DualStudent(
    ff_bias_init=0.5 * FF_BIAS_INIT,
    cdf_bias_init=0.5 * CDF_BIAS_INIT,
).to(DEVICE)
student_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"\nStudent V15 ({BACKBONE_NAME}, 6ch, dual classifiers): {student_params:.2f}M params")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LOSS V15
# ══════════════════════════════════════════════════════════════════════════════

class DualKDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sl, s_feat, sA_feat, sB_feat, fused_feat, gate,
                proj_feats,  # tuple (proj_A, proj_B)
                lA, lB, fA_teacher, fB_teacher,
                y, dom_is_ff, alpha_kd=0.3, beta_kd=0.3):
        eps = 1e-8
        T = KD_TEMPERATURE
        dom_is_cdf = 1.0 - dom_is_ff

        proj_A, proj_B = proj_feats

        # CROSS_DOMAIN_W=0.0 → each teacher only supervises its native domain
        w_A = dom_is_ff  # Teacher A → FF++ samples only
        w_B = dom_is_cdf  # Teacher B → CDF samples only

        # ── Soft KD (T=1.5) ──────────────────────────────────────────────────
        sA_soft = torch.sigmoid(lA.float() / T).detach()
        sB_soft = torch.sigmoid(lB.float() / T).detach()
        sT = sl.float() / T

        L_sA = (w_A * F.binary_cross_entropy_with_logits(
            sT, sA_soft, reduction='none')).mean()
        L_sB = (w_B * F.binary_cross_entropy_with_logits(
            sT, sB_soft, reduction='none')).mean()

        # ── Conflict weighting ────────────────────────────────────────────────
        with torch.no_grad():
            pA = torch.cat([1 - sA_soft, sA_soft], 1).clamp(eps, 1 - eps)
            pB = torch.cat([1 - sB_soft, sB_soft], 1).clamp(eps, 1 - eps)
            kl = (pA * (pA.log() - pB.log())).sum(1, keepdim=True).clamp(min=0)
            cw = 1.0 + LAMBDA_CONFLICT * kl * (dom_is_ff + dom_is_cdf)
            cw = cw / cw.mean()

        # ── Hard focal loss ───────────────────────────────────────────────────
        pw = dom_is_ff * FF_POS_WEIGHT + dom_is_cdf * CDF_POS_WEIGHT
        ys = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING  # = y (ls=0)
        bce_h = F.binary_cross_entropy_with_logits(
            sl.float(), ys, pos_weight=pw, reduction='none')
        focal = ((1 - torch.exp(-bce_h)) ** FOCAL_GAMMA) * bce_h
        L_hard = (cw * focal).mean()

        # ── V15 FIX-5: Feature KD via Cosine Similarity ───────────────────────
        # L_feat = 1 - cos_sim(proj, teacher_feat)
        # Range: 0 (perfect alignment) to 2 (opposite)
        # Much more informative than L2 MSE on near-unit vectors
        if ETA > 0:
            # proj_A aligned to Teacher A (FF++ samples only)
            cos_A = F.cosine_similarity(
                proj_A.float(), fA_teacher.float().detach(), dim=1)
            feat_loss_A = (w_A.squeeze(1) * (1.0 - cos_A)).mean()

            # proj_B aligned to Teacher B (CDF samples only)
            cos_B = F.cosine_similarity(
                proj_B.float(), fB_teacher.float().detach(), dim=1)
            feat_loss_B = (w_B.squeeze(1) * (1.0 - cos_B)).mean()

            L_feat = feat_loss_A + feat_loss_B
        else:
            L_feat = torch.tensor(0.0, device=sl.device)

        # ── Bimodal gate sparsity (encourage gate toward 0 or 1) ─────────────
        gate_e = gate.float().clamp(1e-6, 1 - 1e-6)
        # Entropy: maximised at 0.5 (bad — random routing), minimised near 0/1
        # We MINIMISE entropy → push gate toward bimodal routing
        gate_entropy = -(gate_e * gate_e.log() + (1 - gate_e) * (1 - gate_e).log()).mean()
        L_gate = -gate_entropy  # negative entropy → minimising L_gate maximises bimodality

        # ── Total loss ────────────────────────────────────────────────────────
        total = (GAMMA * L_hard
                 + alpha_kd * L_sA
                 + beta_kd * L_sB
                 + ETA * L_feat
                 + 0.02 * L_gate)

        with torch.no_grad():
            c_h = GAMMA * L_hard.item()
            denom = (c_h + alpha_kd * L_sA.item() + beta_kd * L_sB.item()
                     + ETA * L_feat.item() + 1e-9)
            pct_hard = 100.0 * c_h / denom

        return total, {
            'L_hard': L_hard.item(),
            'L_soft_A': L_sA.item(),
            'L_soft_B': L_sB.item(),
            'L_feat': L_feat.item(),
            'L_gate': L_gate.item(),
            'mean_kl': kl.mean().item(),
            'pct_hard': pct_hard,
        }


criterion = DualKDLoss()


def get_progressive_kd_weights(epoch, total_epochs):
    progress = min(1.0, epoch / total_epochs)
    alpha = ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * progress
    beta = BETA_MAX - (BETA_MAX - BETA_MIN) * progress
    return alpha, beta


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — DOMAIN-AWARE MixUp (V15 CRITICAL FIX)
# ══════════════════════════════════════════════════════════════════════════════

def get_mixup_alpha(epoch):
    """
    V15 schedule:
    - epoch < MIXUP_WARMUP_EPOCHS: no MixUp
    - epoch in [WARMUP, WARMUP+RAMP]: linear ramp-up 0 → MIXUP_ALPHA
    - epoch in [WARMUP+RAMP, EPOCHS-TURNOFF]: full MIXUP_ALPHA
    - epoch > EPOCHS-TURNOFF: no MixUp (clean final convergence)
    """
    if not USE_MIXUP or epoch < MIXUP_WARMUP_EPOCHS:
        return 0.0
    # V15 FIX-2: Turn off for last MIXUP_TURNOFF_EPOCHS epochs
    if epoch > (EPOCHS - MIXUP_TURNOFF_EPOCHS):
        return 0.0
    ramp_ep = epoch - MIXUP_WARMUP_EPOCHS
    if ramp_ep < MIXUP_RAMP_EPOCHS:
        return MIXUP_ALPHA * (ramp_ep + 1) / MIXUP_RAMP_EPOCHS
    return MIXUP_ALPHA


def mixup_ff_only(x, y, dom, epoch):
    alpha = get_mixup_alpha(epoch)
    ff_mask = (dom == 0)  # boolean mask for FF++ samples

    # If no MixUp or no FF++ samples in this batch, return unchanged
    if alpha <= 0 or ff_mask.sum() < 2:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))

    ff_idx = ff_mask.nonzero(as_tuple=True)[0]

    # Mix ONLY within FF++ samples (within-domain, within-batch)
    perm = ff_idx[torch.randperm(len(ff_idx), device=x.device)]
    x_mix = x.clone()
    x_mix[ff_idx] = lam * x[ff_idx] + (1 - lam) * x[perm]

    # Build y_b: same as y for all samples, but permuted partner for FF++ only
    y_b = y.clone()
    y_b[ff_idx] = y[perm]

    # CDF samples: y_a == y_b, lam unused → loss is identical to un-mixed
    return x_mix, y, y_b, lam


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — OPTIMISER + SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

def _get_head_params():
    return (list(student.expert_A.parameters())
            + list(student.expert_B.parameters())
            + list(student.domain_gate.parameters())
            + list(student.classifier_ff.parameters())
            + list(student.classifier_cdf.parameters())
            + list(student.proj_head_A.parameters())
            + list(student.proj_head_B.parameters()))


optimizer = optim.AdamW([
    {'params': list(student.backbone.parameters()), 'lr': LR_BACKBONE},
    {'params': _get_head_params(), 'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)


class CosWarm:
    """Single-cycle cosine annealing with warmup. No restarts."""

    def __init__(self, start_step=0):
        self.cycle = EPOCHS  # Full cycle = EPOCHS (no restarts)
        self.cur = start_step

    def step(self):
        cyc = min(self.cur, self.cycle - 1)
        for g, base in zip(optimizer.param_groups, [LR_BACKBONE, LR_HEAD]):
            mn = base * 0.01
            if cyc < WARMUP_EPOCHS:
                lr = mn + (base - mn) * (cyc + 1) / WARMUP_EPOCHS
            else:
                p = (cyc - WARMUP_EPOCHS) / max(self.cycle - WARMUP_EPOCHS, 1)
                lr = mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * p))
            g['lr'] = lr
        self.cur += 1

    def lrs(self):
        return [g['lr'] for g in optimizer.param_groups]


scaler = torch.cuda.amp.GradScaler(enabled=_use_amp)
swa_model = AveragedModel(student)

HKEYS = ['epoch', 'train_loss', 'train_acc',
         'val_ff_acc', 'val_cdf_acc', 'val_combined_acc',
         'val_ff_auc', 'val_cdf_auc',  # V15: AUC tracked for early stop
         'lr_backbone', 'lr_head',
         'L_hard', 'L_soft_A', 'L_soft_B', 'L_feat', 'L_gate',
         'mean_kl', 'pct_hard', 'steps_this_epoch',
         'alpha_kd', 'beta_kd', 'mixup_alpha']
history = {k: [] for k in HKEYS}
scheduler = CosWarm(start_step=0)
best_val_acc = 0.0
best_ff_auc = 0.0
best_cdf_auc = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def savefig(name):
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches='tight')
    plt.close();
    print(f"  📊 {name}")


_val_preds_ff = {'y': None, 'p': None}
_val_preds_cdf = {'y': None, 'p': None}


def val_pass(loader, desc, store_key=None):
    """Returns (loss, accuracy, auc). AUC requires storing predictions."""
    student.eval()
    loss_sum = correct = total = 0
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y, dom, _ in tqdm(loader, desc=desc, leave=False):
            x = x.to(DEVICE);
            y = y.unsqueeze(1).to(DEVICE)
            dom_ids = dom.to(DEVICE)
            dom_is_ff = (dom == 0).float().unsqueeze(1).to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=_use_amp):
                fA, lA = teacher_A.get_features_and_logit(x[:, :3])
                fB, lB = teacher_B.get_features_and_logit(x[:, :3])
                logit, s_f, sA_f, sB_f, fused_f, gate, proj_fs = \
                    student.forward_full(x, dom_ids)
                alpha_kd, beta_kd = get_progressive_kd_weights(
                    len(history['epoch']), EPOCHS)
                loss, _ = criterion(logit, s_f, sA_f, sB_f, fused_f, gate,
                                    proj_fs, lA, lB, fA, fB, y, dom_is_ff,
                                    alpha_kd=alpha_kd, beta_kd=beta_kd)
            loss_sum += loss.item()
            probs = torch.sigmoid(logit).squeeze(1)
            correct += ((probs >= 0.5).float() == y.squeeze(1)).sum().item()
            total += y.size(0)
            all_y.extend(y.squeeze(1).cpu().numpy().tolist())
            all_p.extend(probs.cpu().float().numpy().tolist())

    all_y = np.array(all_y);
    all_p = np.array(all_p)

    # Store for threshold/calibration fitting
    if store_key == 'ff':
        _val_preds_ff['y'] = all_y;
        _val_preds_ff['p'] = all_p
    elif store_key == 'cdf':
        _val_preds_cdf['y'] = all_y;
        _val_preds_cdf['p'] = all_p

    # Compute AUC (used for V15 per-domain early stopping)
    auc_val = float('nan')
    if len(np.unique(all_y)) == 2:
        try:
            auc_val = roc_auc_score(all_y, all_p)
        except:
            pass

    return loss_sum / max(len(loader), 1), correct / max(total, 1), auc_val


def get_best_thresh(store_key):
    d = _val_preds_ff if store_key == 'ff' else _val_preds_cdf
    if d['y'] is None or len(np.unique(d['y'])) < 2:
        return 0.5
    try:
        fpr_, tpr_, th_ = roc_curve(d['y'], d['p'])
        return float(np.clip(th_[np.argmax(tpr_ - fpr_)], 0.1, 0.9))
    except:
        return 0.5


def infer(model, loader, desc, tta=False, domain_id=None):
    model.eval()
    ys, ps, doms, idxs, lats = [], [], [], [], []
    mean_rgb = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std_rgb = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    def dn_rgb(x):
        return x[:, :3] * std_rgb + mean_rgb

    def rn_rgb(x):
        return (x[:, :3] - mean_rgb) / std_rgb

    def flip_6ch(x):
        return torch.cat([torch.flip(x[:, :3], [3]), torch.flip(x[:, 3:], [3])], dim=1)

    def bright_6ch(x):
        return torch.cat([rn_rgb(torch.clamp(dn_rgb(x) * 1.05, 0, 1)), x[:, 3:]], dim=1)

    fns = [lambda x: x, flip_6ch, bright_6ch]

    with torch.no_grad():
        for x, y, dom, idx in tqdm(loader, desc=desc, leave=False):
            x = x.to(DEVICE)
            t0 = time.perf_counter()
            if domain_id is not None:
                dom_ids = torch.full((x.size(0),), domain_id, dtype=torch.long, device=DEVICE)
            else:
                dom_ids = dom.to(DEVICE)

            if tta:
                ls = None
                for fn in fns:
                    with torch.amp.autocast(DEVICE.type, enabled=_use_amp):
                        l = model(fn(x), domain_ids=dom_ids) \
                            if hasattr(model, 'forward_full') else model(fn(x[:, :3]))
                    ls = l if ls is None else ls + l
                logit = ls / len(fns)
            else:
                with torch.amp.autocast(DEVICE.type, enabled=_use_amp):
                    if hasattr(model, 'forward_full'):
                        logit = model(x, domain_ids=dom_ids)
                    else:
                        logit = model(x[:, :3])  # teachers: RGB only

            lats.append((time.perf_counter() - t0) / x.size(0) * 1000)
            ps.extend(torch.sigmoid(logit).squeeze(1).cpu().float().numpy().tolist())
            ys.extend(y.numpy().tolist())
            doms.extend(dom.numpy().tolist())
            idxs.extend(idx.numpy().tolist())
    return np.array(ys), np.array(ps), np.array(doms), np.array(idxs), float(np.mean(lats))


def infer_teacher(model, loader, desc):
    """Teachers only see RGB (3ch)."""
    model.eval()
    ys, ps, doms, idxs, lats = [], [], [], [], []
    with torch.no_grad():
        for x, y, dom, idx in tqdm(loader, desc=desc, leave=False):
            x_rgb = x[:, :3].to(DEVICE)
            t0 = time.perf_counter()
            with torch.amp.autocast(DEVICE.type, enabled=_use_amp):
                logit = model(x_rgb)
            lats.append((time.perf_counter() - t0) / x.size(0) * 1000)
            ps.extend(torch.sigmoid(logit).squeeze(1).cpu().float().numpy().tolist())
            ys.extend(y.numpy().tolist())
            doms.extend(dom.numpy().tolist())
            idxs.extend(idx.numpy().tolist())
    return np.array(ys), np.array(ps), np.array(doms), np.array(idxs), float(np.mean(lats))


def compute_metrics(y, p, d, name, lat):
    if len(np.unique(y)) < 2:
        return {'Model': name, 'AUC_ROC': float('nan'),
                'Accuracy': accuracy_score(y, d), 'Latency_ms': lat}
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    fpr_, tpr_, _ = roc_curve(y, p);
    pr_, rc_, _ = precision_recall_curve(y, p)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y, d),
        'Balanced_Acc': balanced_accuracy_score(y, d),
        'Precision': precision_score(y, d, zero_division=0),
        'Recall': recall_score(y, d, zero_division=0),
        'F1_Fake': f1_score(y, d, pos_label=1, zero_division=0),
        'AUC_ROC': auc(fpr_, tpr_),
        'AUC_PR': auc(rc_, pr_),
        'MCC': matthews_corrcoef(y, d),
        'Cohen_Kappa': cohen_kappa_score(y, d),
        'Brier': brier_score_loss(y, p),
        'Log_Loss': log_loss(y, p),
        'FPR': fp / (fp + tn + 1e-9),
        'FNR': fn / (fn + tp + 1e-9),
        'Specificity': tn / (tn + fp + 1e-9),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'Params_M': student_params,
        'Latency_ms': lat,
    }


def ece(y, p, n=10):
    bins = np.linspace(0, 1, n + 1);
    e = 0.0
    for i in range(n):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum(): e += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return e


def save_ckpt(ep, acc, tag="", emergency=False):
    name = f"{'EMERGENCY_' if emergency else ''}dual_teacher_v15_epoch{ep:02d}{tag}.pth"
    obj = {'epoch': ep, 'state_dict': student.state_dict(),
           'optimizer': optimizer.state_dict(), 'val_acc': acc, 'history': history}
    torch.save(obj, CKPT_DIR / name);
    torch.save(obj, OUT_DIR / name)
    print(f"  💾 {name}")


def flush_history():
    pd.DataFrame(history).to_csv(
        OUT_DIR / 'dual_teacher_v15_training_history.csv', index=False)


def _fmt(v, fmt='.4f'):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'nan'
    return format(v, fmt)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TEMPERATURE SCALING (V15 FIX-4: skip if T near 1)
# ══════════════════════════════════════════════════════════════════════════════

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor([0.0]))

    @property
    def T(self):
        return torch.exp(self.log_T).clamp(0.5, 5.0)

    def forward(self, logits):
        return logits / self.T

    def fit(self, epochs_done):
        if epochs_done < 10:
            print(f"  ⚠ Skipping temp scaling (only {epochs_done} epochs — need ≥10)")
            self.log_T.data = torch.zeros(1)
            return 1.0, False

        all_logits, all_labels = [], []
        for store_d in [_val_preds_ff, _val_preds_cdf]:
            if store_d['y'] is not None and store_d['p'] is not None:
                p_clamped = np.clip(store_d['p'], 1e-6, 1 - 1e-6)
                logit_vals = np.log(p_clamped / (1 - p_clamped))
                all_logits.append(
                    torch.tensor(logit_vals, dtype=torch.float32).unsqueeze(1))
                all_labels.append(
                    torch.tensor(store_d['y'], dtype=torch.float32).unsqueeze(1))

        if not all_logits:
            print("  ⚠ No val predictions for temp scaling — T=1.0")
            self.log_T.data = torch.zeros(1)
            return 1.0, False

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        probs_before = torch.sigmoid(logits).numpy().ravel()
        ece_before = ece(labels.numpy().ravel(), probs_before)

        self.log_T.data = torch.tensor([math.log(1.5)])
        ts_opt = optim.Adam([self.log_T], lr=0.05)
        best_ece = float('inf')
        best_log_T = self.log_T.data.clone()

        try:
            for _ in range(200):
                ts_opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(logits / self.T, labels)
                loss.backward()
                ts_opt.step()
                with torch.no_grad():
                    cur_probs = torch.sigmoid(logits / self.T).numpy().ravel()
                    cur_ece = ece(labels.numpy().ravel(), cur_probs)
                    if cur_ece < best_ece:
                        best_ece = cur_ece
                        best_log_T = self.log_T.data.clone()

            self.log_T.data = best_log_T
            T_v = self.T.item()
            probs_after = torch.sigmoid(logits / self.T).detach().numpy().ravel()
            ece_after = ece(labels.numpy().ravel(), probs_after)

            # V15 FIX-4: Skip calibration if T is essentially a no-op
            if 0.8 <= T_v <= 1.2:
                print(f"  ⚠ T={T_v:.4f} ∈ [0.8, 1.2] — calibration not meaningful; "
                      f"using T=1.0 (ECE before: {ece_before:.4f})")
                self.log_T.data = torch.zeros(1)
                return 1.0, False

            # Also skip if calibration makes ECE worse
            if ece_after > ece_before * 1.05:
                print(f"  ⚠ Calibration hurt ECE ({ece_before:.4f}→{ece_after:.4f}), T={T_v:.4f} — using T=1.0")
                self.log_T.data = torch.zeros(1)
                return 1.0, False

            print(f"  ✅ T={T_v:.4f}  ECE: {ece_before:.4f}→{ece_after:.4f}")
            return T_v, True

        except Exception as e:
            print(f"  ⚠ Temp scaling failed ({e}) — T=1.0")
            self.log_T.data = torch.zeros(1)
            return 1.0, False


temp_scaler = TemperatureScaling()

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — TRAINING LOOP V15
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print(f"  DUAL-TEACHER KD V15  |  ep 1–{EPOCHS}  |  {DEVICE}")
print(f"  CRITICAL FIX: Domain-aware MixUp — CDF NEVER gets MixUp")
print(f"  Evidence: val_cdf_acc 0.892→0.136 exactly when MixUp activated (ep 10)")
print(f"  Other fixes: cosine feat KD, ALPHA_MIN=0.15, T skip if ∈[0.8,1.2]")
print(f"{'═' * 70}\n")

best_epoch = 0
no_improve = 0
stop_reason = None
ep_times = []

# V15 FIX-8: Per-domain patience tracking
no_improve_ff = 0
no_improve_cdf = 0

try:
    for epoch in range(1, EPOCHS + 1):

        est = np.mean(ep_times[-3:]) if ep_times else 40 * 60
        if time_left() < est + EVAL_RESERVE:
            stop_reason = f"time_limit (tl={time_left() / 60:.1f}min)"
            print(f"\n⏰  {stop_reason}");
            break

        student.train()
        scheduler.step()
        lrs = scheduler.lrs()
        ep_t0 = time.time()

        alpha_kd, beta_kd = get_progressive_kd_weights(epoch, EPOCHS)
        cur_mixup_alpha = get_mixup_alpha(epoch)
        mixup_active = cur_mixup_alpha > 0

        rl = rh = rsA = rsB = rfeat = rgate = rkl = rpct = 0.0
        correct = total = step_count = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", total=effective_steps)
        for batch_idx, (x, y, dom) in enumerate(pbar):
            if MAX_STEPS_PER_EPOCH and batch_idx >= MAX_STEPS_PER_EPOCH: break
            if time_left() < EVAL_RESERVE:
                stop_reason = "mid_epoch_time_limit"
                save_ckpt(epoch, best_val_acc, tag="_PARTIAL", emergency=True)
                flush_history();
                break

            x = x.to(DEVICE);
            y = y.unsqueeze(1).to(DEVICE)
            dom_ids = dom.to(DEVICE)
            dom_is_ff = (dom == 0).float().unsqueeze(1).to(DEVICE)

            # V15 FIX-1: Domain-aware MixUp — CDF samples are NEVER mixed
            x_mix, y_a, y_b, lam = mixup_ff_only(x, y, dom_ids, epoch)

            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE.type, enabled=_use_amp):
                # Teachers only see RGB channels
                with torch.no_grad():
                    fA_t, lA = teacher_A.get_features_and_logit(x_mix[:, :3])
                    fB_t, lB = teacher_B.get_features_and_logit(x_mix[:, :3])
                logit, s_f, sA_f, sB_f, fused_f, gate, proj_fs = \
                    student.forward_full(x_mix, dom_ids)

                loss_a, parts = criterion(
                    logit, s_f, sA_f, sB_f, fused_f, gate, proj_fs,
                    lA, lB, fA_t, fB_t, y_a, dom_is_ff,
                    alpha_kd=alpha_kd, beta_kd=beta_kd)

                # MixUp interpolated loss (only relevant for FF++ samples)
                if lam < 0.999:
                    loss_b, _ = criterion(
                        logit, s_f, sA_f, sB_f, fused_f, gate, proj_fs,
                        lA, lB, fA_t, fB_t, y_b, dom_is_ff,
                        alpha_kd=alpha_kd, beta_kd=beta_kd)
                    # For CDF samples: y_a == y_b, so loss_a == loss_b
                    # For FF++ samples: lam*loss_a + (1-lam)*loss_b
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = loss_a

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            scaler.step(optimizer);
            scaler.update()

            step_count += 1;
            rl += loss.item();
            rh += parts['L_hard']
            rsA += parts['L_soft_A'];
            rsB += parts['L_soft_B']
            rfeat += parts['L_feat'];
            rgate += parts['L_gate']
            rkl += parts['mean_kl'];
            rpct += parts['pct_hard']
            correct += ((torch.sigmoid(logit) >= 0.5).float() == y_a).sum().item()
            total += y_a.size(0)

            pbar.set_postfix(
                loss=f"{rl / step_count:.4f}", acc=f"{correct / max(total, 1):.4f}",
                hard=f"{rh / step_count:.3f}", pct=f"{rpct / step_count:.0f}%",
                sA=f"{rsA / step_count:.3f}",
                feat=f"{rfeat / step_count:.3f}",
                αkd=f"{alpha_kd:.2f}",
                mx=f"{'FF✓' if mixup_active else '✗'}{cur_mixup_alpha:.2f}",
                tl=f"{time_left() / 60:.0f}m")

        ep_times.append(time.time() - ep_t0)
        if stop_reason: break

        # Validation — track both accuracy AND AUC
        _, vff, vff_auc = val_pass(ff_val_loader, f"Ep{epoch} val-FF", store_key='ff')
        _, vcdf, vcdf_auc = val_pass(cdf_val_loader, f"Ep{epoch} val-CDF", store_key='cdf')
        vc = 0.5 * (vff + vcdf)

        # SWA update
        if epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(student)

        nb = max(step_count, 1)
        for k, v in [
            ('epoch', epoch), ('train_loss', rl / nb), ('train_acc', correct / max(total, 1)),
            ('val_ff_acc', vff), ('val_cdf_acc', vcdf), ('val_combined_acc', vc),
            ('val_ff_auc', vff_auc), ('val_cdf_auc', vcdf_auc),
            ('lr_backbone', lrs[0]), ('lr_head', lrs[1]),
            ('L_hard', rh / nb), ('L_soft_A', rsA / nb), ('L_soft_B', rsB / nb),
            ('L_feat', rfeat / nb), ('L_gate', rgate / nb),
            ('mean_kl', rkl / nb), ('pct_hard', rpct / nb),
            ('steps_this_epoch', step_count),
            ('alpha_kd', alpha_kd), ('beta_kd', beta_kd),
            ('mixup_alpha', cur_mixup_alpha),
        ]:
            history[k].append(v)

        is_best = vc > best_val_acc
        save_ckpt(epoch, vc, tag="_BEST" if is_best else "");
        flush_history()

        if is_best:
            best_val_acc = vc;
            best_epoch = epoch;
            no_improve = 0
            if not math.isnan(vff_auc):  best_ff_auc = vff_auc
            if not math.isnan(vcdf_auc): best_cdf_auc = vcdf_auc
            torch.save(student.state_dict(),
                       OUT_DIR / 'best_dual_teacher_v15_student.pth')
            print(f"  🌟 new best={vc:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                stop_reason = "plateau";
                break

        ep_min = ep_times[-1] / 60
        auc_str = (f"auc_ff={_fmt(vff_auc)} auc_cdf={_fmt(vcdf_auc)}"
                   if not math.isnan(vff_auc) else "")
        print(f"  Ep{epoch:02d} tr={correct / max(total, 1):.4f} "
              f"vFF={vff:.4f} vCDF={vcdf:.4f} vc={vc:.4f} | "
              f"{auc_str} | "
              f"hard={rh / nb:.3f}({rpct / nb:.0f}%) sA={rsA / nb:.3f} "
              f"feat={rfeat / nb:.3f} | "
              f"αkd={alpha_kd:.2f} mx={'FF✓' if mixup_active else '✗'}{cur_mixup_alpha:.2f} | "
              f"ep={ep_min:.1f}m tl={time_left() / 60:.0f}m"
              + ("  🌟" if is_best else ""))
        print("─" * 70)
        gc.collect();
        torch.cuda.empty_cache()

except Exception as exc:
    stop_reason = f"EXCEPTION: {exc}"
    print(f"\n💥 {stop_reason}");
    traceback.print_exc()
    try:
        ep_crash = history['epoch'][-1] if history['epoch'] else 0
        save_ckpt(ep_crash, best_val_acc, tag="_CRASH", emergency=True)
        flush_history()
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — POST-TRAINING EVAL
# ══════════════════════════════════════════════════════════════════════════════

finally:
    print(f"\n{'═' * 70}")
    print(f"  POST-TRAINING | stop: {stop_reason or 'completed normally'}")
    print(f"  best_epoch={best_epoch}  best_val_acc={best_val_acc:.4f}")
    print(f"  best_ff_auc={_fmt(best_ff_auc)}  best_cdf_auc={_fmt(best_cdf_auc)}")
    print(f"  time_left: {time_left() / 60:.1f} min")
    if ep_times: print(f"  avg epoch time: {np.mean(ep_times) / 60:.1f} min")
    print(f"{'═' * 70}")

    epochs_done = len(history['epoch'])

    # ── SWA — V15 FIX-3: Start at epoch 20 (after clean training) ─────────
    swa_ready = epochs_done >= SWA_START_EPOCH
    print(f"\n[SWA] {'→ finalising' if swa_ready else f'→ SKIPPED (epochs {epochs_done} < {SWA_START_EPOCH})'}")
    if swa_ready:
        try:
            update_bn(train_loader, swa_model, device=DEVICE)
            torch.save(swa_model.module.state_dict(),
                       OUT_DIR / 'swa_dual_teacher_v15_student.pth')
            print("  💾 swa saved")
        except Exception as e:
            print(f"  ⚠ SWA: {e}");
            swa_ready = False

    # ── Load best checkpoint ──────────────────────────────────────────────
    bp = OUT_DIR / 'best_dual_teacher_v15_student.pth'
    if bp.exists():
        try:
            student.load_state_dict(torch.load(bp, map_location=DEVICE, weights_only=True))
            print("  ✅ best model loaded")
        except Exception as e:
            try:
                raw = torch.load(bp, map_location=DEVICE, weights_only=False)
                sd = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
                student.load_state_dict(sd, strict=False)
                print("  ✅ best model loaded (strict=False)")
            except Exception as e2:
                print(f"  ⚠ Best load failed: {e2}")
    else:
        ckpts = sorted(CKPT_DIR.glob("dual_teacher_v15_epoch*.pth"))
        if ckpts:
            try:
                raw = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
                sd = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
                student.load_state_dict(sd, strict=False)
                print(f"  ✅ fallback: {ckpts[-1].name}")
            except Exception as e:
                print(f"  ⚠ Fallback failed: {e}")

    # ── Temperature scaling — V15 FIX-4 ──────────────────────────────────
    print(f"\n[TEMP SCALING] Fitting (skip if T ∈ [0.8, 1.2]) …")
    T_val = 1.0
    cal_was_applied = False
    try:
        T_val, cal_was_applied = temp_scaler.fit(epochs_done)
        torch.save(temp_scaler.state_dict(), OUT_DIR / 'calibrated_temp_scale_v15.pth')
    except Exception as e:
        print(f"  ⚠ TS error: {e}");
        traceback.print_exc()

    T_FF = get_best_thresh('ff')
    T_CDF = get_best_thresh('cdf')
    print(f"\n  Thresholds: FF++={T_FF:.4f}  CDF={T_CDF:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  EVAL INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    R = {}
    ff_preds = None
    cdf_preds = None


    def _swa_valid(p):
        return (p is not None and not np.all(p == 0.0) and np.std(p) > 0.01)


    def save_predictions_now(tag, T_threshold):
        pred_dict = ff_preds if tag == 'ff' else cdf_preds
        if pred_dict is None: return
        n = len(pred_dict['true_label'])
        prob_std = pred_dict.get('prob_std', np.zeros(n))
        prob_tta = pred_dict.get('prob_tta', np.zeros(n))
        prob_swa = pred_dict.get('prob_swa', np.zeros(n))
        prob_cal = pred_dict.get('prob_cal', np.zeros(n))
        swa_ok = _swa_valid(prob_swa)

        if 'prob_tta' in pred_dict and swa_ok:
            prob_ens = 0.4 * prob_std + 0.4 * prob_tta + 0.2 * prob_swa
        elif 'prob_tta' in pred_dict:
            prob_ens = 0.5 * prob_std + 0.5 * prob_tta
        else:
            prob_ens = prob_std

        df = pd.DataFrame({
            'true_label': pred_dict['true_label'].astype(int),
            'prob_std': np.round(prob_std, 6),
            'prob_tta': np.round(prob_tta, 6),
            'prob_swa': np.round(prob_swa if swa_ok else np.zeros(n), 6),
            'prob_cal': np.round(prob_cal, 6),
            'prob_ensemble': np.round(prob_ens, 6),
            'pred_std': (prob_std >= T_threshold).astype(int),
            'pred_tta': (prob_tta >= T_threshold).astype(int),
            'pred_ensemble': (prob_ens >= T_threshold).astype(int),
        })
        df.to_csv(OUT_DIR / f'{tag}_test_predictions_v15.csv', index=False)
        populated = [k for k in ['prob_std', 'prob_tta', 'prob_swa', 'prob_cal']
                     if k in pred_dict and (k != 'prob_swa' or swa_ok)]
        print(f"  💾 {tag}_predictions_v15.csv  populated={populated}  swa_valid={swa_ok}")


    def run_infer_required(ky, kp, kl, ki, model, loader, desc,
                           tta=False, domain_id=None, is_teacher=False):
        try:
            if is_teacher:
                y, p, doms, idxs, lat = infer_teacher(model, loader, desc)
            else:
                y, p, doms, idxs, lat = infer(model, loader, desc,
                                              tta=tta, domain_id=domain_id)
            R[ky] = y;
            R[kp] = p;
            R[kl] = lat
            if ki: R[ki] = idxs
            if ky + '_dom' not in R: R[ky + '_dom'] = doms
            astr = f"AUC={roc_auc_score(y, p):.4f}" if len(np.unique(y)) > 1 else ""
            print(f"  ✅ {desc}  {astr}  tl={time_left() / 60:.1f}min")
        except Exception as e:
            print(f"  ❌ {desc} FAILED: {e}");
            traceback.print_exc()


    def run_infer_budgeted(ky, kp, kl, ki, model, loader, desc, budget_secs,
                           tta=False, domain_id=None, is_teacher=False):
        if time_left() < budget_secs:
            print(f"  ⏰ SKIP {desc} (tl={time_left() / 60:.1f}m < {budget_secs // 60}m)")
            return
        run_infer_required(ky, kp, kl, ki, model, loader, desc,
                           tta=tta, domain_id=domain_id, is_teacher=is_teacher)


    # Phase 1: Student std
    print("\n[EVAL Phase 1] Student std …")
    run_infer_required('ff_y', 'ff_p', 'ff_lat', 'ff_idx',
                       student, ff_test_loader, "FF++ Std", domain_id=0)
    run_infer_required('cdf_y', 'cdf_p', 'cdf_lat', 'cdf_idx',
                       student, cdf_test_loader, "CDF Std", domain_id=1)

    if 'ff_y' in R: ff_preds = {'true_label': R['ff_y'], 'prob_std': R['ff_p']}
    if 'cdf_y' in R: cdf_preds = {'true_label': R['cdf_y'], 'prob_std': R['cdf_p']}
    save_predictions_now('ff', T_FF)
    save_predictions_now('cdf', T_CDF)

    # Phase 2: Teachers
    print(f"\n[EVAL Phase 2] Teachers …")
    run_infer_budgeted('ta_ff_y', 'ta_ff_p', 'ta_ff_lat', None,
                       teacher_A, ff_test_loader, "Teacher-A→FF++", EVAL_BUDGET_TEACHER, is_teacher=True)
    run_infer_budgeted('tb_cdf_y', 'tb_cdf_p', 'tb_cdf_lat', None,
                       teacher_B, cdf_test_loader, "Teacher-B→CDF", EVAL_BUDGET_TEACHER, is_teacher=True)
    run_infer_budgeted('ta_cdf_y', 'ta_cdf_p', 'ta_cdf_lat', None,
                       teacher_A, cdf_test_loader, "Teacher-A→CDF", EVAL_BUDGET_TEACHER, is_teacher=True)
    run_infer_budgeted('tb_ff_y', 'tb_ff_p', 'tb_ff_lat', None,
                       teacher_B, ff_test_loader, "Teacher-B→FF++", EVAL_BUDGET_TEACHER, is_teacher=True)

    # Phase 3: TTA
    print(f"\n[EVAL Phase 3] TTA …")
    run_infer_budgeted('ff_y', 'ff_p_tta', 'ff_lat_tta', None,
                       student, ff_test_loader, "FF++ TTA", EVAL_BUDGET_OPTIONAL, tta=True, domain_id=0)
    run_infer_budgeted('cdf_y', 'cdf_p_tta', 'cdf_lat_tta', None,
                       student, cdf_test_loader, "CDF TTA", EVAL_BUDGET_OPTIONAL, tta=True, domain_id=1)
    if ff_preds is not None and 'ff_p_tta' in R:
        ff_preds['prob_tta'] = R['ff_p_tta'];
        save_predictions_now('ff', T_FF)
    if cdf_preds is not None and 'cdf_p_tta' in R:
        cdf_preds['prob_tta'] = R['cdf_p_tta'];
        save_predictions_now('cdf', T_CDF)

    # Phase 3b: SWA
    if swa_ready:
        print(f"\n[EVAL Phase 3b] SWA …")
        run_infer_budgeted('ff_y', 'ff_p_swa', 'ff_lat_swa', None,
                           swa_model.module, ff_test_loader,
                           "FF++ SWA", EVAL_BUDGET_OPTIONAL, domain_id=0)
        run_infer_budgeted('cdf_y', 'cdf_p_swa', 'cdf_lat_swa', None,
                           swa_model.module, cdf_test_loader,
                           "CDF SWA", EVAL_BUDGET_OPTIONAL, domain_id=1)
        if ff_preds is not None and 'ff_p_swa' in R:
            ff_preds['prob_swa'] = R['ff_p_swa'];
            save_predictions_now('ff', T_FF)
        if cdf_preds is not None and 'cdf_p_swa' in R:
            cdf_preds['prob_swa'] = R['cdf_p_swa'];
            save_predictions_now('cdf', T_CDF)
    else:
        print(f"\n[SWA SKIPPED] epochs_done={epochs_done} < {SWA_START_EPOCH}")

    if cal_was_applied:
        try:
            for k in ['ff_p', 'ff_p_tta', 'ff_p_swa', 'cdf_p', 'cdf_p_tta', 'cdf_p_swa']:
                if k in R and R[k] is not None:
                    lt = torch.logit(torch.tensor(R[k]).float().clamp(1e-6, 1 - 1e-6))
                    with torch.no_grad():
                        R[k + '_cal'] = torch.sigmoid(temp_scaler(lt)).detach().numpy()
            print(f"  ✅ Calibrated (T={T_val:.4f})")
            best_ff_cal = next((k for k in ['ff_p_tta_cal', 'ff_p_swa_cal', 'ff_p_cal'] if k in R), None)
            best_cdf_cal = next((k for k in ['cdf_p_tta_cal', 'cdf_p_swa_cal', 'cdf_p_cal'] if k in R), None)
            if ff_preds is not None and best_ff_cal:
                ff_preds['prob_cal'] = R[best_ff_cal];
                save_predictions_now('ff', T_FF)
            if cdf_preds is not None and best_cdf_cal:
                cdf_preds['prob_cal'] = R[best_cdf_cal];
                save_predictions_now('cdf', T_CDF)
        except Exception as e:
            print(f"  ⚠ cal: {e}");
            traceback.print_exc()
    else:
        print(f"  ℹ Calibration skipped (T={T_val:.4f} ∈ [0.8, 1.2] or not beneficial)")
        # Still copy prob_tta as prob_cal for consistency in output CSVs
        if ff_preds is not None and 'ff_p_tta' in R:
            ff_preds['prob_cal'] = R['ff_p_tta'];
            save_predictions_now('ff', T_FF)
        if cdf_preds is not None and 'cdf_p_tta' in R:
            cdf_preds['prob_cal'] = R['cdf_p_tta'];
            save_predictions_now('cdf', T_CDF)

    # ECE computation
    ECE_FF = ECE_CDF = ECE_FF_CAL = ECE_CDF_CAL = float('nan')
    if 'ff_y' in R and 'ff_p_tta' in R: ECE_FF = ece(R['ff_y'], R['ff_p_tta'])
    if 'cdf_y' in R and 'cdf_p_tta' in R: ECE_CDF = ece(R['cdf_y'], R['cdf_p_tta'])
    if 'ff_y' in R and 'ff_p_tta_cal' in R: ECE_FF_CAL = ece(R['ff_y'], R['ff_p_tta_cal'])
    if 'cdf_y' in R and 'cdf_p_tta_cal' in R: ECE_CDF_CAL = ece(R['cdf_y'], R['cdf_p_tta_cal'])

    # Metrics
    M = {}


    def try_metrics(key, yk, pk, lk, name, thresh=0.5):
        if yk in R and pk in R and lk in R:
            try:
                M[key] = compute_metrics(
                    R[yk], R[pk], (R[pk] >= thresh).astype(int), name, R[lk])
            except Exception as e:
                print(f"  ⚠ metrics {name}: {e}")


    try_metrics('ff_std', 'ff_y', 'ff_p', 'ff_lat', "Student-FF-Std", T_FF)
    try_metrics('ff_tta', 'ff_y', 'ff_p_tta', 'ff_lat_tta', "Student-FF-TTA", T_FF)
    try_metrics('ff_swa', 'ff_y', 'ff_p_swa', 'ff_lat_swa', "Student-FF-SWA", T_FF)
    try_metrics('ff_cal', 'ff_y', 'ff_p_tta_cal', 'ff_lat_tta', "Student-FF-Cal", T_FF)
    try_metrics('cdf_std', 'cdf_y', 'cdf_p', 'cdf_lat', "Student-CDF-Std", T_CDF)
    try_metrics('cdf_tta', 'cdf_y', 'cdf_p_tta', 'cdf_lat_tta', "Student-CDF-TTA", T_CDF)
    try_metrics('cdf_swa', 'cdf_y', 'cdf_p_swa', 'cdf_lat_swa', "Student-CDF-SWA", T_CDF)
    try_metrics('cdf_cal', 'cdf_y', 'cdf_p_tta_cal', 'cdf_lat_tta', "Student-CDF-Cal", T_CDF)
    try_metrics('ta_ff', 'ta_ff_y', 'ta_ff_p', 'ta_ff_lat', "Teacher-A-FF++", 0.5)
    try_metrics('tb_cdf', 'tb_cdf_y', 'tb_cdf_p', 'tb_cdf_lat', "Teacher-B-CDF", 0.5)
    try_metrics('ta_cdf', 'ta_cdf_y', 'ta_cdf_p', 'ta_cdf_lat', "Teacher-A→CDF", 0.5)
    try_metrics('tb_ff', 'tb_ff_y', 'tb_ff_p', 'tb_ff_lat', "Teacher-B→FF++", 0.5)

    # Ensemble metrics
    for tag, yk, pk_std, pk_tta, pk_swa, th in [
        ('ff', 'ff_y', 'ff_p', 'ff_p_tta', 'ff_p_swa', T_FF),
        ('cdf', 'cdf_y', 'cdf_p', 'cdf_p_tta', 'cdf_p_swa', T_CDF),
    ]:
        if yk not in R or pk_std not in R: continue
        swa_v = _swa_valid(R.get(pk_swa))
        if pk_tta in R and swa_v:
            ens = 0.4 * R[pk_std] + 0.4 * R[pk_tta] + 0.2 * R[pk_swa]
        elif pk_tta in R:
            ens = 0.5 * R[pk_std] + 0.5 * R[pk_tta]
        else:
            ens = R[pk_std]
        R[f'{tag}_p_ens'] = ens
        try_metrics(f'{tag}_ens', yk, f'{tag}_p_ens', f'{tag}_lat',
                    f"Student-{tag.upper()}-Ensemble", th)

    for section, keys in [
        ("FF++", ['ff_std', 'ff_tta', 'ff_swa', 'ff_cal', 'ff_ens', 'ta_ff', 'tb_ff']),
        ("CDF", ['cdf_std', 'cdf_tta', 'cdf_swa', 'cdf_cal', 'cdf_ens', 'tb_cdf', 'ta_cdf']),
    ]:
        print(f"\n── {section} ──")
        for k in keys:
            if k in M:
                m = M[k]
                print(f"  {m['Model']:<35} Acc={_fmt(m.get('Accuracy'))}  "
                      f"AUC={_fmt(m.get('AUC_ROC'))}  F1={_fmt(m.get('F1_Fake'))}  "
                      f"MCC={_fmt(m.get('MCC'))}")

    print(f"\n  ECE (T={T_val:.4f}, applied={cal_was_applied}): "
          f"FF++ {_fmt(ECE_FF)}→{_fmt(ECE_FF_CAL)}  "
          f"CDF {_fmt(ECE_CDF)}→{_fmt(ECE_CDF_CAL)}")

    # Phase 5: Per-manipulation analysis
    print(f"\n[EVAL Phase 5] Per-manipulation + Bootstrap …")
    manip_df = pd.DataFrame()
    try:
        if 'ff_y' in R and 'ff_idx' in R:
            pk_use = next((k for k in ['ff_p_ens', 'ff_p_swa', 'ff_p_tta', 'ff_p'] if k in R), None)
            if pk_use:
                sub_indices = ff_te_sub.indices
                sample_pos = R['ff_idx'].astype(int)
                orig_indices = [sub_indices[i] for i in sample_pos]
                manip_arr = np.array([ff_full.idx_to_manip[i] for i in orig_indices])
                y_all = R['ff_y'];
                p_all = R[pk_use]
                real_idx = np.where(y_all == 0)[0]
                rng_m = np.random.default_rng(SEED);
                mres = []
                for m in sorted(set(manip_arr)):
                    fake_idx = np.where((manip_arr == m) & (y_all == 1))[0]
                    if m == 'real' or len(fake_idx) == 0:
                        ri = np.where(manip_arr == 'real')[0]
                        ym = y_all[ri];
                        pm = p_all[ri];
                        dm = (pm >= T_FF).astype(int)
                        mres.append({'ManipType': 'real', 'Count': int(len(ri)),
                                     'Real_Paired': '—', 'AUC_ROC': float('nan'),
                                     'Accuracy': round(accuracy_score(ym, dm), 4),
                                     'Note': 'Accuracy=TNR'});
                        continue
                    n_f = len(fake_idx)
                    sr = rng_m.choice(real_idx, min(n_f, len(real_idx)), replace=False)
                    ym = y_all[np.concatenate([fake_idx, sr])]
                    pm = p_all[np.concatenate([fake_idx, sr])]
                    dm = (pm >= T_FF).astype(int)
                    row = {'ManipType': m, 'Count': int(n_f), 'Real_Paired': int(len(sr))}
                    if len(np.unique(ym)) > 1:
                        fpr_m, tpr_m, _ = roc_curve(ym, pm)
                        row.update({
                            'AUC_ROC': round(auc(fpr_m, tpr_m), 4),
                            'Accuracy': round(accuracy_score(ym, dm), 4),
                            'F1_Fake': round(f1_score(ym, dm, pos_label=1, zero_division=0), 4),
                            'MCC': round(matthews_corrcoef(ym, dm), 4),
                            'Precision': round(precision_score(ym, dm, zero_division=0), 4),
                            'Recall': round(recall_score(ym, dm, zero_division=0), 4),
                        })
                    mres.append(row)
                manip_df = pd.DataFrame(mres)
                manip_df.to_csv(OUT_DIR / 'per_manipulation_metrics_v15.csv', index=False)
                print("\n── Per-manipulation (V15) ──")
                print(manip_df.to_string(index=False))
    except Exception as e:
        print(f"  ⚠ per-manip: {e}")

    if time_left() > EVAL_BUDGET_EXTRAS:
        print("[BOOTSTRAP] n=500 …")
        try:
            rng = np.random.default_rng(SEED)
            mfns = {
                'AUC-ROC': lambda y, p, d: roc_auc_score(y, p),
                'Accuracy': lambda y, p, d: accuracy_score(y, d),
                'F1-Fake': lambda y, p, d: f1_score(y, d, zero_division=0),
                'MCC': lambda y, p, d: matthews_corrcoef(y, d),
            }
            boot_sets = []
            for tag, yk, pk, th in [
                ("FF++ Ens", 'ff_y', 'ff_p_ens', T_FF),
                ("CDF Ens", 'cdf_y', 'cdf_p_ens', T_CDF),
                ("FF++ SWA", 'ff_y', 'ff_p_swa', T_FF),
                ("CDF SWA", 'cdf_y', 'cdf_p_swa', T_CDF),
                ("FF++ TTA", 'ff_y', 'ff_p_tta', T_FF),
                ("CDF TTA", 'cdf_y', 'cdf_p_tta', T_CDF),
                ("FF++ Std", 'ff_y', 'ff_p', T_FF),
                ("CDF Std", 'cdf_y', 'cdf_p', T_CDF),
            ]:
                if yk in R and pk in R and _swa_valid(R.get(pk)):
                    boot_sets.append((tag, R[yk], R[pk], (R[pk] >= th).astype(int)))
                elif yk in R and pk in R and 'swa' not in pk.lower():
                    boot_sets.append((tag, R[yk], R[pk], (R[pk] >= th).astype(int)))
            rows = []
            for tag, y, p, d in boot_sets:
                for mn, fn in mfns.items():
                    idx = np.arange(len(y));
                    s = []
                    for _ in range(500):
                        b = rng.choice(idx, len(idx), replace=True)
                        try:
                            s.append(fn(y[b], p[b], d[b]))
                        except:
                            pass
                    s = np.array(s)
                    rows.append({'Dataset': tag, 'Metric': mn,
                                 'Mean': round(s.mean(), 4),
                                 'CI_lo': round(np.percentile(s, 2.5), 4),
                                 'CI_hi': round(np.percentile(s, 97.5), 4)})
            pd.DataFrame(rows).to_csv(OUT_DIR / 'dual_teacher_v15_bootstrap_ci.csv', index=False)
            print(pd.DataFrame(rows).to_string(index=False))
        except Exception as e:
            print(f"  ⚠ Bootstrap: {e}")

    try:
        pd.DataFrame(list(M.values())).to_csv(
            OUT_DIR / 'dual_teacher_v15_metrics.csv', index=False)
        print("  ✅ metrics.csv")
    except Exception as e:
        print(f"  ⚠ metrics.csv: {e}")
    flush_history();
    print("  ✅ history.csv")

    # ── Figures ───────────────────────────────────────────────────────────
    if time_left() > EVAL_BUDGET_EXTRAS:
        print("\n[FIGURES] …")
        plt.rcParams.update({
            'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
            'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
            'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
            'grid.color': '#21262d', 'grid.linestyle': '--', 'grid.alpha': 0.6,
            'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
            'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d',
            'savefig.facecolor': '#0d1117', 'savefig.dpi': 200,
        })
        CB = '#58a6ff';
        CR = '#f85149';
        CG = '#3fb950';
        CY = '#d29922'
        CP = '#bc8cff';
        CGR = '#484f58';
        CO = '#ffa657';
        CT = '#39d0d8'
        ep_x = history['epoch']

        try:  # FIG 01 — training overview
            fig, axes = plt.subplots(2, 2, figsize=(18, 10))

            axes[0, 0].plot(ep_x, history['train_loss'], color=CB, lw=2.5, marker='o', ms=3)
            axes[0, 0].set_title('Total Loss');
            axes[0, 0].grid(True)

            axes[0, 1].plot(ep_x, history['train_acc'], color=CG, lw=2.5, label='Train')
            axes[0, 1].plot(ep_x, history['val_ff_acc'], color=CB, lw=2.5, ls='--', label='Val FF++')
            axes[0, 1].plot(ep_x, history['val_cdf_acc'], color=CT, lw=2.5, ls='-.', label='Val CDF')
            axes[0, 1].plot(ep_x, history['val_combined_acc'], color=CY, lw=2.5, ls=':', label='Val Combined')
            axes[0, 1].axhline(0.5, color=CR, lw=1, ls=':', alpha=0.4, label='Random')
            # Mark MixUp window
            mx_start = MIXUP_WARMUP_EPOCHS
            mx_end = EPOCHS - MIXUP_TURNOFF_EPOCHS
            if mx_start <= max(ep_x, default=0):
                axes[0, 1].axvspan(mx_start, min(mx_end, max(ep_x, default=mx_end)),
                                   alpha=0.08, color=CY, label=f'MixUp window (FF++ only)')
            axes[0, 1].set_title('Accuracy — V15 (CDF protected from MixUp)')
            axes[0, 1].legend(fontsize=8);
            axes[0, 1].grid(True)

            axes[1, 0].plot(ep_x, history['L_hard'], color=CG, lw=2, label='L_hard')
            axes[1, 0].plot(ep_x, history['L_soft_A'], color=CB, lw=2, ls='--', label='L_sA (FF++→student)')
            axes[1, 0].plot(ep_x, history['L_soft_B'], color=CT, lw=2, ls=':', label='L_sB (CDF→student)')
            axes[1, 0].plot(ep_x, history['L_feat'], color=CO, lw=2, ls='-.', label=f'L_feat (cosine, ETA={ETA})')
            axes[1, 0].axhline(0.25, color=CY, lw=1, ls=':', alpha=0.5, label='KD target <0.25')
            axes[1, 0].set_title('Loss Components V15');
            axes[1, 0].legend(fontsize=8);
            axes[1, 0].grid(True)

            if history.get('val_ff_auc'):
                axes[1, 1].plot(ep_x, history['val_ff_auc'], color=CB, lw=2.5, label='Val FF++ AUC')
                axes[1, 1].plot(ep_x, history['val_cdf_auc'], color=CT, lw=2.5, ls='--', label='Val CDF AUC')
                axes[1, 1].plot(ep_x, history['alpha_kd'], color=CO, lw=1.5, ls='-.', alpha=0.8, label='alpha_kd')
                axes[1, 1].plot(ep_x, history['mixup_alpha'], color=CY, lw=1.5, ls=':', alpha=0.8,
                                label='MixUp α (FF++ only)')
                axes[1, 1].axhline(0.92, color=CB, lw=1, ls=':', alpha=0.4, label='FF++ target 0.92')
                axes[1, 1].axhline(0.96, color=CT, lw=1, ls=':', alpha=0.4, label='CDF target 0.96')
                axes[1, 1].set_ylim(0.0, 1.02)
                axes[1, 1].set_title('Val AUC + KD Schedule V15');
                axes[1, 1].legend(fontsize=8);
                axes[1, 1].grid(True)

            fig.suptitle('Dual-Teacher KD V15 — Training Overview\n(CDF protected from MixUp)',
                         fontsize=14)
            fig.tight_layout();
            savefig('fig01_training_v15.png')
        except Exception as e:
            print(f"  ⚠ fig01: {e}")

        try:  # FIG 02 — version comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            categories = ['FF++ AUC (Std)', 'CDF AUC (Std)', 'FF++ AUC (SWA)', 'CDF AUC (SWA)']
            v13_vals = [0.8264, 0.9422, float('nan'), float('nan')]
            v14_vals = [0.9076, 0.9755, 0.9382, 0.9841]
            v15_vals = [
                M.get('ff_std', {}).get('AUC_ROC', float('nan')),
                M.get('cdf_std', {}).get('AUC_ROC', float('nan')),
                M.get('ff_swa', {}).get('AUC_ROC', float('nan')),
                M.get('cdf_swa', {}).get('AUC_ROC', float('nan')),
            ]
            targets = [0.92, 0.96, 0.93, 0.97]
            x = np.arange(len(categories));
            w = 0.25
            for i, (vers_vals, col, lbl) in enumerate([
                (v13_vals, CGR, 'V13'), (v14_vals, CY, 'V14'), (v15_vals, CG, 'V15')]):
                vals = [v if not (isinstance(v, float) and math.isnan(v)) else 0 for v in vers_vals]
                axes[0].bar(x + (i - 1) * w, vals, w, color=col, alpha=0.85, label=lbl)
            for xi, tv in enumerate(targets):
                axes[0].plot([xi - 1.5 * w, xi + 1.5 * w], [tv, tv], color=CR, lw=1.5, ls='--', alpha=0.6)
            axes[0].set_xticks(x);
            axes[0].set_xticklabels(categories, fontsize=9, rotation=10)
            axes[0].set_ylim(0.78, 1.01);
            axes[0].set_ylabel('AUC-ROC')
            axes[0].set_title('V13→V14→V15 (red dashed = target)')
            axes[0].legend(fontsize=9);
            axes[0].grid(axis='y', alpha=0.4)

            ens_keys = [('ff_swa', 'FF++ SWA ★'), ('cdf_swa', 'CDF SWA ★'),
                        ('ff_ens', 'FF++ Ens'), ('cdf_ens', 'CDF Ens'),
                        ('ff_std', 'FF++ Std'), ('cdf_std', 'CDF Std')]
            names_, aucs_ = [], []
            for k, lbl in ens_keys:
                if k in M and not math.isnan(M[k].get('AUC_ROC', float('nan'))):
                    names_.append(lbl);
                    aucs_.append(M[k]['AUC_ROC'])
            if names_:
                colors_ = [CG if 'SWA' in n else (CB if 'Ens' in n else CGR) for n in names_]
                bars = axes[1].bar(names_, aucs_, color=colors_, alpha=0.85)
                axes[1].set_ylim(min(aucs_) - 0.02 if aucs_ else 0.9, 1.01)
                axes[1].set_title('V15 All Models (AUC-ROC)')
                axes[1].grid(axis='y', alpha=0.4)
                axes[1].axhline(0.92, color=CB, lw=1, ls='--', alpha=0.5)
                axes[1].axhline(0.96, color=CT, lw=1, ls='--', alpha=0.5)
                for bar, v in zip(bars, aucs_):
                    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                                 f'{v:.4f}', ha='center', fontsize=9, color='#c9d1d9')
            fig.suptitle('V15 Results Overview', fontsize=13);
            fig.tight_layout()
            savefig('fig02_v15_results.png')
        except Exception as e:
            print(f"  ⚠ fig02: {e}")

        try:  # FIG 03 — ROC curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, yk, pk, pk_tta, pk_swa, pk_ens, ta_k, title in [
                (axes[0], 'ff_y', 'ff_p', 'ff_p_tta', 'ff_p_swa', 'ff_p_ens', 'ta_ff_p', 'FF++ Test'),
                (axes[1], 'cdf_y', 'cdf_p', 'cdf_p_tta', 'cdf_p_swa', 'cdf_p_ens', 'tb_cdf_p', 'CDF Test'),
            ]:
                if yk not in R: continue
                for pk_, col, ls_, lbl in [
                    (pk, CB, '-', 'Std'),
                    (pk_tta, CT, '--', 'TTA'),
                    (pk_swa, CO, '-.', 'SWA ★'),
                    (pk_ens, CG, '-', 'Ensemble ★'),
                    (ta_k, CR, ':', 'Teacher'),
                ]:
                    if pk_ in R and R[pk_] is not None and _swa_valid(R.get(pk_)):
                        try:
                            f_, t_, _ = roc_curve(R[yk], R[pk_])
                            ax.plot(f_, t_, color=col, lw=2.5 if '★' in lbl else 2, ls=ls_,
                                    label=f"{lbl} AUC={roc_auc_score(R[yk], R[pk_]):.4f}")
                        except:
                            pass
                    elif pk_ in R and 'swa' not in str(pk_).lower():
                        try:
                            f_, t_, _ = roc_curve(R[yk], R[pk_])
                            ax.plot(f_, t_, color=col, lw=2, ls=ls_,
                                    label=f"{lbl} AUC={roc_auc_score(R[yk], R[pk_]):.4f}")
                        except:
                            pass
                ax.plot([0, 1], [0, 1], color=CGR, lw=1, ls=':')
                ax.set_title(title);
                ax.legend(fontsize=9);
                ax.grid(True)
            fig.suptitle('ROC Curves V15', fontsize=13);
            fig.tight_layout()
            savefig('fig03_roc_v15.png')
        except Exception as e:
            print(f"  ⚠ fig03: {e}")

        try:  # FIG 04 — score distributions (critical: check CDF separation)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ri, (yk, pk, title, thresh) in enumerate([
                ('ff_y', 'ff_p_tta' if 'ff_p_tta' in R else 'ff_p', 'FF++', T_FF),
                ('cdf_y', 'cdf_p_tta' if 'cdf_p_tta' in R else 'cdf_p', 'CDF', T_CDF),
            ]):
                if yk not in R or pk not in R: continue
                y_ = R[yk];
                p_ = R[pk];
                d_ = (p_ >= thresh).astype(int)
                axes[ri, 0].hist(p_[y_ == 0], bins=60, color=CG, alpha=0.7, label='REAL', density=True)
                axes[ri, 0].hist(p_[y_ == 1], bins=60, color=CR, alpha=0.7, label='FAKE', density=True)
                axes[ri, 0].axvline(thresh, color=CY, lw=2, ls='--', label=f'τ={thresh:.3f}')
                axes[ri, 0].legend();
                axes[ri, 0].grid(True)
                axes[ri, 0].set_title(f'Score Distribution — {title} '
                                      f'(max={p_.max():.3f} min={p_.min():.3f})')
                cm_ = confusion_matrix(y_, d_, labels=[0, 1]).astype(float)
                cm_n = cm_ / cm_.sum(1, keepdims=True)
                sns.heatmap(cm_n, annot=True, fmt='.3f', ax=axes[ri, 1], cmap='Blues',
                            xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
                            annot_kws={'size': 13, 'weight': 'bold'})
                axes[ri, 1].set_title(f'Confusion Matrix — {title}')
            fig.tight_layout();
            savefig('fig04_distributions_v15.png')
        except Exception as e:
            print(f"  ⚠ fig04: {e}")

        try:  # FIG 05 — feature KD convergence (key diagnostic for V15)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            if history.get('L_feat'):
                axes[0].plot(ep_x, history['L_soft_A'], color=CB, lw=2,
                             label='L_sA (target <0.25)')
                axes[0].plot(ep_x, history['L_soft_B'], color=CT, lw=2, ls='--',
                             label='L_sB (target <0.25)')
                axes[0].plot(ep_x, history['L_feat'], color=CO, lw=2.5, ls='-.',
                             label=f'L_feat cosine (ETA={ETA}) — should be meaningful')
                axes[0].axhline(0.25, color=CY, lw=1.5, ls=':', label='<0.25 target')
                axes[0].set_title('KD Convergence V15 (cosine feat KD vs V14 MSE≈0)')
                axes[0].legend(fontsize=9);
                axes[0].grid(True)

            if history.get('mixup_alpha'):
                # Overlay MixUp schedule alongside val accuracies — crucial diagnostic
                ax2 = axes[1].twinx()
                axes[1].plot(ep_x, history['val_ff_acc'], color=CB, lw=2.5, label='Val FF++ Acc')
                axes[1].plot(ep_x, history['val_cdf_acc'], color=CT, lw=2.5, ls='--', label='Val CDF Acc')
                ax2.plot(ep_x, history['mixup_alpha'], color=CY, lw=1.5, ls=':', alpha=0.8,
                         label='MixUp alpha (FF++ only)')
                axes[1].set_ylabel('Accuracy');
                ax2.set_ylabel('MixUp alpha')
                axes[1].set_ylim(0.0, 1.05)
                axes[1].axhline(0.5, color=CR, lw=1, ls=':', alpha=0.4)
                axes[1].set_title('Val Acc vs MixUp Schedule\n(CDF should stay stable when MixUp activates)')
                axes[1].legend(fontsize=9, loc='lower left')
                ax2.legend(fontsize=9, loc='upper right')
                axes[1].grid(True)
            fig.suptitle('V15 Feature KD + MixUp Diagnostics', fontsize=13)
            fig.tight_layout();
            savefig('fig05_feat_kd_v15.png')
        except Exception as e:
            print(f"  ⚠ fig05: {e}")


    # ── Final summary ─────────────────────────────────────────────────────
    def g(mk, field):
        v = M.get(mk, {}).get(field, float('nan'))
        return v if v is not None else float('nan')


    avg_ep = np.mean(ep_times) / 60 if ep_times else float('nan')
    pct_h = history['pct_hard'][-1] if history.get('pct_hard') else float('nan')
    l_sA_final = history['L_soft_A'][-1] if history.get('L_soft_A') else float('nan')
    l_feat_final = history['L_feat'][-1] if history.get('L_feat') else float('nan')
    final_alpha = history['alpha_kd'][-1] if history.get('alpha_kd') else float('nan')
    kd_converged = not math.isnan(l_sA_final) and l_sA_final < 0.25

    swa_line = ""
    if swa_ready:
        swa_line = (f"\n  ── Student SWA ──\n"
                    f"  FF++ AUC={_fmt(g('ff_swa', 'AUC_ROC'))}  Acc={_fmt(g('ff_swa', 'Accuracy'))}"
                    f"  F1={_fmt(g('ff_swa', 'F1_Fake'))}  MCC={_fmt(g('ff_swa', 'MCC'))}\n"
                    f"  CDF  AUC={_fmt(g('cdf_swa', 'AUC_ROC'))}  Acc={_fmt(g('cdf_swa', 'Accuracy'))}"
                    f"  F1={_fmt(g('cdf_swa', 'F1_Fake'))}  MCC={_fmt(g('cdf_swa', 'MCC'))}\n")

    # Check if MixUp disrupted CDF (key V15 diagnostic)
    h_cdf = history.get('val_cdf_acc', [])
    if len(h_cdf) >= MIXUP_WARMUP_EPOCHS + 1:
        pre_mix_cdf = h_cdf[MIXUP_WARMUP_EPOCHS - 1]  # epoch just before MixUp
        post_mix_cdf = h_cdf[MIXUP_WARMUP_EPOCHS]  # epoch when MixUp starts
        mixup_drop = pre_mix_cdf - post_mix_cdf
        mixup_ok = mixup_drop < 0.05
        mixup_status = f"{'✅ CDF stable (+{-mixup_drop:.3f})' if mixup_ok else f'⚠ CDF dropped {mixup_drop:.3f}'}"
    else:
        mixup_status = "N/A (not enough epochs)"

    summary = (
            f"\n  DUAL-TEACHER KD V15 — FINAL SUMMARY\n"
            f"  Stop reason     : {stop_reason or 'completed normally'}\n"
            f"  Epochs trained  : {epochs_done}\n"
            f"  Best epoch      : {best_epoch}\n"
            f"  Best val FF AUC : {_fmt(best_ff_auc)}\n"
            f"  Best val CDF AUC: {_fmt(best_cdf_auc)}\n"
            f"  Avg ep time     : {_fmt(avg_ep, '.1f')} min\n"
            f"  Hard loss %     : {_fmt(pct_h, '.1f')}%\n"
            f"  L_soft_A final  : {_fmt(l_sA_final)} "
            f"({'✅ converged' if kd_converged else '⚠ still high'})\n"
            f"  L_feat final    : {_fmt(l_feat_final)} (cosine, target ~0.3-0.8)\n"
            f"  Final alpha_kd  : {_fmt(final_alpha, '.3f')}\n\n"
            f"  V15 MixUp diagnostic (key fix):\n"
            f"  {mixup_status}\n\n"
            f"  V15 FIXES APPLIED:\n"
            f"  [FIX-1] Domain-aware MixUp: CDF NEVER mixed (was destroying CDF at ep10)\n"
            f"  [FIX-2] MixUp OFF last {MIXUP_TURNOFF_EPOCHS} eps → clean final convergence\n"
            f"  [FIX-3] SWA_START={SWA_START_EPOCH} → clean post-training averaging\n"
            f"  [FIX-4] Skip calibration if T∈[0.8,1.2] (T=1.103 hurt ECE in V14)\n"
            f"  [FIX-5] Cosine feat KD (ETA=1.0) — L_feat was ≈0.0005 with MSE\n"
            f"  [FIX-6] ALPHA_MIN=0.15 (was 0.05 — teacher signal sustained)\n"
            f"  [FIX-7] Simple cosine decay, no restarts\n"
            f"  [FIX-9] CDF_POS_WEIGHT capped at 3.0 (not 5.0)\n\n"
            f"  ── Student Std ──\n"
            f"  FF++ AUC={_fmt(g('ff_std', 'AUC_ROC'))}  Acc={_fmt(g('ff_std', 'Accuracy'))}"
            f"  F1={_fmt(g('ff_std', 'F1_Fake'))}  MCC={_fmt(g('ff_std', 'MCC'))}\n"
            f"  CDF  AUC={_fmt(g('cdf_std', 'AUC_ROC'))}  Acc={_fmt(g('cdf_std', 'Accuracy'))}"
            f"  F1={_fmt(g('cdf_std', 'F1_Fake'))}  MCC={_fmt(g('cdf_std', 'MCC'))}\n\n"
            f"  ── Student TTA ──\n"
            f"  FF++ AUC={_fmt(g('ff_tta', 'AUC_ROC'))}  Acc={_fmt(g('ff_tta', 'Accuracy'))}"
            f"  F1={_fmt(g('ff_tta', 'F1_Fake'))}  MCC={_fmt(g('ff_tta', 'MCC'))}\n"
            f"  CDF  AUC={_fmt(g('cdf_tta', 'AUC_ROC'))}  Acc={_fmt(g('cdf_tta', 'Accuracy'))}"
            f"  F1={_fmt(g('cdf_tta', 'F1_Fake'))}  MCC={_fmt(g('cdf_tta', 'MCC'))}\n\n"
            f"  ── Ensemble ──\n"
            f"  FF++ AUC={_fmt(g('ff_ens', 'AUC_ROC'))}  Acc={_fmt(g('ff_ens', 'Accuracy'))}"
            f"  F1={_fmt(g('ff_ens', 'F1_Fake'))}  MCC={_fmt(g('ff_ens', 'MCC'))}\n"
            f"  CDF  AUC={_fmt(g('cdf_ens', 'AUC_ROC'))}  Acc={_fmt(g('cdf_ens', 'Accuracy'))}"
            f"  F1={_fmt(g('cdf_ens', 'F1_Fake'))}  MCC={_fmt(g('cdf_ens', 'MCC'))}\n"
            + swa_line +
            f"\n  ── Calibration (T={T_val:.4f}, applied={cal_was_applied}) ──\n"
            f"  FF++ ECE: {_fmt(ECE_FF)}→{_fmt(ECE_FF_CAL)}\n"
            f"  CDF  ECE: {_fmt(ECE_CDF)}→{_fmt(ECE_CDF_CAL)}\n\n"
            f"  ── Teacher Baselines ──\n"
            f"  Teacher A FF++ (native)  AUC={_fmt(g('ta_ff', 'AUC_ROC'))}\n"
            f"  Teacher A→CDF  (cross)   AUC={_fmt(g('ta_cdf', 'AUC_ROC'))}\n"
            f"  Teacher B CDF  (native)  AUC={_fmt(g('tb_cdf', 'AUC_ROC'))}\n"
            f"  Teacher B→FF++ (cross)   AUC={_fmt(g('tb_ff', 'AUC_ROC'))}\n\n"
            f"  ── Efficiency ──\n"
            f"  {student_params:.2f}M params ({BACKBONE_NAME}, 6ch, dual classifiers)\n"
            f"  Latency={_fmt(g('ff_std', 'Latency_ms'), '.2f')}ms/frame\n"
            f"  Elapsed: {elapsed() / 3600:.2f}h\n\n"
            f"  ── V15 Targets ──\n"
            f"  FF++ AUC (Std) target: >0.92  | Achieved: {_fmt(g('ff_std', 'AUC_ROC'))}\n"
            f"  CDF  AUC (Std) target: >0.96  | Achieved: {_fmt(g('cdf_std', 'AUC_ROC'))}\n"
            f"  FF++ AUC (SWA) target: >0.93  | Achieved: {_fmt(g('ff_swa', 'AUC_ROC'))}\n"
            f"  CDF  AUC (SWA) target: >0.97  | Achieved: {_fmt(g('cdf_swa', 'AUC_ROC'))}\n"
    )
    print(f"\n{'═' * 70}")
    print(summary)
    try:
        (OUT_DIR / 'final_summary_v15.txt').write_text(summary)
    except:
        pass
    print(f"{'═' * 70}")