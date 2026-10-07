import torch
import torch.nn.functional as F

_identity_cache = {}

def get_identity_matrix(n, device):
    key = (n, str(device))
    if key not in _identity_cache:
        _identity_cache[key] = torch.eye(n, device=device)
    return _identity_cache[key]


#  Patch decorrelation loss  (Eq. 1) 

def patch_decorrelation_loss(attention_hooks):
    """
    Args:
        attention_hooks: list of AttentionMap hooks, one per layer

    Returns:
       loss_pd
    """
    loss_pd = 0.0
    for hook in attention_hooks:
        feature = hook.feature                     # (B, heads, N, head_dim)
        feature_p = feature.mean(dim=1)[:, 1:, :]  # head-average, drop CLS row
        feature_p = F.normalize(feature_p.float(), dim=-1)

        gram = torch.bmm(feature_p, feature_p.transpose(1, 2))   # (B, P, P)
        identity = get_identity_matrix(gram.size(1), gram.device).unsqueeze(0)
        loss_pd = loss_pd + ((gram - identity) ** 2).mean(dim=(1, 2)).mean()
    return loss_pd


#  Image decorrelation loss  (Eq. 2)

def image_decorrelation_loss(global_features):
    """   
    Args:
        global_features: final-layer CLS embeddings (or GAP of patch
        embeddings for teachers without a CLS token), shape (B, C)

    Returns:
        loss_id
    """

    global_features = F.normalize(global_features, dim=-1)
    gram = global_features @ global_features.T                    # (B, B)
    identity = get_identity_matrix(gram.size(0), gram.device)
    loss_id = ((gram - identity) ** 2).mean()

    return loss_id


#  TV Loss (L^TV). Taken from PSAQ-ViT codebase (see github.com/zkkli/PSAQ-ViT/generate_data.py)
def tv_loss(images):

    diff1 = images[:, :, :, :-1] - images[:, :, :, 1:]
    diff2 = images[:, :, :-1, :] - images[:, :, 1:, :]
    diff3 = images[:, :, 1:, :-1] - images[:, :, :-1, 1:]
    diff4 = images[:, :, :-1, :-1] - images[:, :, 1:, 1:]

    loss = (torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4))
    return loss
