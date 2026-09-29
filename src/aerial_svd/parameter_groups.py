"""Parameter report categories shared by both runtimes."""
def classify_param(name: str) -> str:
    if "attn.qkv.weight" in name:
        return "attention_qkv"
    if "attn.proj.weight" in name:
        return "attention_proj"
    if "mlp.fc1.weight" in name:
        return "ffn_expand"
    if "mlp.fc2.weight" in name:
        return "ffn_contract"
    if "patch_embed.proj.weight" in name:
        return "patch_embed"
    if name.endswith("head.weight"):
        return "classifier"
    if "norm" in name:
        return "normalization"
    if "bias" in name:
        return "bias"
    if "pos_embed" in name:
        return "positional_embedding"
    if "cls_token" in name:
        return "class_token"
    return "other"

