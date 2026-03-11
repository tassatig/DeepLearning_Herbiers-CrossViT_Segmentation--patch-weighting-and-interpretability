import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1) Capture attentions self-attention d'une branche CrossViT
#    crossvit_O2.py: VisionTransformer.blocks = [MultiScaleBlock]
#    MultiScaleBlock.blocks[branch] = nn.Sequential(timm Block, ...)
# ============================================================

class CrossViTBranchAttentionCapture:
    """
    Capture (B, H, N, N) par couche pour une branche CrossViT.
    Hook sur blk.attn.attn_drop : inputs[0] = attn après softmax (timm).
    """
    def __init__(self, crossvit_backbone: nn.Module, branch_id: int = 0, detach: bool = False):
        self.m = crossvit_backbone
        self.branch_id = branch_id
        self.detach = detach
        self.attns = []
        self.handles = []

    def _hook_fn(self, module, inputs, output):
        a = inputs[0]
        if torch.is_tensor(a):
            self.attns.append(a.detach() if self.detach else a)

    def start(self):
        self.attns = []
        self.handles = []

        for msb in self.m.blocks:  # VisionTransformer.blocks :contentReference[oaicite:3]{index=3}
            if not hasattr(msb, "blocks") or msb.blocks is None:
                continue
            seq = msb.blocks[self.branch_id]  # nn.Sequential de timm Block :contentReference[oaicite:4]{index=4}
            for blk in seq:
                if not (hasattr(blk, "attn") and hasattr(blk.attn, "attn_drop")):
                    raise ValueError("Structure inattendue: attn_drop introuvable dans un block.")
                h = blk.attn.attn_drop.register_forward_hook(self._hook_fn)
                self.handles.append(h)

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


# ============================================================
# 2) Attention rollout -> heatmap CLS->patches
# ============================================================

def attention_rollout(attn_list, add_identity=True, eps=1e-6):
    """
    attn_list: list of (B, H, N, N)
    returns: (B, N, N)
    """
    if len(attn_list) == 0:
        raise ValueError("Aucune attention capturée (attn_list vide).")

    A = [a.mean(dim=1) for a in attn_list]  # (B,N,N)
    B, N, _ = A[0].shape
    eye = torch.eye(N, device=A[0].device, dtype=A[0].dtype).unsqueeze(0).expand(B, -1, -1)

    R = eye
    for a in A:
        if add_identity:
            a = a + eye
        a = a / (a.sum(dim=-1, keepdim=True) + eps)
        R = a @ R
    return R

def cls_rollout_to_patch_heatmap(rollout, grid_hw):
    """
    rollout: (B,N,N)
    grid_hw: (gh,gw)
    returns: (B,1,gh,gw) normalized [0,1]
    """
    gh, gw = grid_hw
    cls_to_patches = rollout[:, 0, 1:]               # (B,N-1)
    heat = cls_to_patches.reshape(-1, 1, gh, gw)     # (B,1,gh,gw)

    flat = heat.flatten(2)
    mn = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
    mx = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    return (heat - mn) / (mx - mn + 1e-6)

def crossvit_grid_hw(crossvit_backbone: nn.Module, branch_id: int):
    """
    crossvit_O2.py: img_size[branch] et patch_embed[branch].patch_size :contentReference[oaicite:5]{index=5}
    """
    img = crossvit_backbone.img_size[branch_id]
    p = crossvit_backbone.patch_embed[branch_id].patch_size[0]
    return (img // p, img // p)


# ============================================================
# 3) Soft IoU loss (différentiable) + wrapper "CE + IoU"
# ============================================================

def soft_iou_loss(att_map, plant_mask, eps=1e-6):
    """
    att_map: (B,1,H,W) in [0,1]
    plant_mask: (B,1,H,W) 0/1
    """
    att = att_map.clamp(0, 1)
    m = plant_mask.float().clamp(0, 1)

    inter = (att * m).flatten(1).sum(dim=1)
    union = (att + m - att * m).flatten(1).sum(dim=1)
    return (1.0 - inter / (union + eps)).mean()

def ce_plus_iou_crossvit(
    model_wrapper,          # ton CrossViTDualInput(...)
    crossvit_backbone,      # model_wrapper.m (VisionTransformer crossvit_O2.py)
    x_small, x_large,       # 2 entrées
    y,
    plant_mask,             # (B,1,H,W) masque plante aligné image
    criterion_ce,
    lambda_iou=0.1,
    branch_id=0,            # branche sur laquelle on calcule la heatmap
):
    if plant_mask.dim() == 3:
        plant_mask = plant_mask.unsqueeze(1)

    with CrossViTBranchAttentionCapture(crossvit_backbone, branch_id=branch_id, detach=False) as cap:
        logits = model_wrapper(x_small, x_large)

    loss_ce = criterion_ce(logits, y)

    rollout = attention_rollout(cap.attns)
    heat_patch = cls_rollout_to_patch_heatmap(rollout, crossvit_grid_hw(crossvit_backbone, branch_id))
    heat_img = F.interpolate(heat_patch, size=plant_mask.shape[-2:], mode="bilinear", align_corners=False)

    loss_iou = soft_iou_loss(heat_img, plant_mask)
    loss_total = loss_ce + lambda_iou * loss_iou

    return loss_total, loss_ce, loss_iou, heat_img, logits
