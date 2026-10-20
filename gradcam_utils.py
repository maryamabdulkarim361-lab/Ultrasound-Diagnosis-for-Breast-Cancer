"""
Grad-CAM helper: generates a heatmap overlay showing which parts of an
ultrasound image most influenced the model's prediction.

Requires the 'grad-cam' package (pip install grad-cam). If it's not
installed, GRADCAM_AVAILABLE is False and generate_gradcam_overlay() returns
None instead of raising -- callers should check GRADCAM_AVAILABLE first, or
handle a None return, so the rest of the app keeps working without it.
"""

import numpy as np

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


def get_target_layer(model, arch):
    """
    Pick the last convolutional layer to visualize, per architecture.
    """
    if arch in ("resnet18", "resnet50"):
        return [model.layer4[-1]]
    elif arch == "efficientnet_b2":
        return [model.features[-1]]
    raise ValueError(f"No known Grad-CAM target layer for arch '{arch}'")


def generate_gradcam_overlay(model, arch, input_tensor, rgb_img_0_1, target_category=None):
    """
    model: trained PyTorch model (eval mode)
    arch: architecture string ('resnet18', 'resnet50', 'efficientnet_b2')
    input_tensor: preprocessed/normalized tensor, shape (1, 3, 224, 224)
    rgb_img_0_1: the same image, as a plain RGB numpy array in [0, 1],
                 shape (224, 224, 3) -- used as the background for the overlay
    target_category: class index to explain (defaults to the model's top prediction)

    Returns: uint8 RGB numpy array (the heatmap overlaid on the image), or
             None if the grad-cam library isn't installed.
    """
    if not GRADCAM_AVAILABLE:
        return None

    target_layers = get_target_layer(model, arch)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img_0_1.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization
