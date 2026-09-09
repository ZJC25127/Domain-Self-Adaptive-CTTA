"""Load the single source ViT used by the MoE experiments."""

from robustbench.model_zoo.our_vit import create_model


def load_source_model():
    return create_model("vit_base_patch16_224", pretrained=True)
