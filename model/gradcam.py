"""
AquaScan - Real Grad-CAM Explainability Module  (Person A deliverable)
=======================================================================
Generates Class Activation Maps using the trained EfficientNetB0 model.
Falls back to the Gaussian simulation in eval.py if weights are missing.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

_GRADCAM_LAYER = "top_conv"


def compute_gradcam(image: Image.Image, model, class_idx=None, layer_name=_GRADCAM_LAYER):
    """
    Real Grad-CAM using gradient signals through the network.

    Args:
        image: PIL Image (any size - resized to 224x224 internally).
        model: Loaded Keras model.
        class_idx: Class to explain. None => argmax of predictions.
        layer_name: Conv layer to hook (default: top_conv).

    Returns:
        heatmap_img (PIL Image), overlay_img (PIL Image)
    """
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    import matplotlib.pyplot as plt

    img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    x = preprocess_input(img_array.copy())
    x = np.expand_dims(x, axis=0)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    from PIL import Image as PILImage
    hmap_pil = PILImage.fromarray(np.uint8(heatmap_np * 255)).resize((224, 224), PILImage.LANCZOS)
    heatmap_resized = np.array(hmap_pil) / 255.0

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_resized)[:, :, :3]

    img_norm = np.array(img, dtype=np.float32) / 255.0
    overlay = 0.45 * colored + 0.55 * img_norm
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    colored_uint8 = np.clip(colored * 255, 0, 255).astype(np.uint8)

    return PILImage.fromarray(colored_uint8), PILImage.fromarray(overlay)


def explain_prediction(image: Image.Image, target_class=None):
    """
    High-level helper for Streamlit.
    Uses real Grad-CAM if weights loaded; falls back to simulation otherwise.
    """
    from model.predict import _load_model, _preprocess, CLASSES
    import numpy as np

    model = _load_model()

    if model is None:
        from model.eval import generate_gradcam_simulation
        return generate_gradcam_simulation(image)

    x = _preprocess(image)
    if target_class is not None:
        if isinstance(target_class, str) and target_class in CLASSES:
            class_idx = CLASSES.index(target_class)
        elif isinstance(target_class, int):
            class_idx = target_class
        else:
            class_idx = None
    else:
        preds = model.predict(x, verbose=0)[0]
        class_idx = int(np.argmax(preds))

    return compute_gradcam(image, model, class_idx=class_idx)