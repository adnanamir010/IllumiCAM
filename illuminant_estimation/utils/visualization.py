import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam_heatmap(model, target_layer, input_tensor, target_category=None, use_cuda=False):
    """
    Generate Grad-CAM heatmap for a specific input and target category.
    """
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
    
    if target_category is not None:
        targets = [ClassifierOutputTarget(target_category)]
    else:
        targets = None # Uses the highest scoring category

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def visualize_heatmap(image_path, heatmap, alpha=0.5, save_path=None):
    """
    Overlay heatmap on original image and save/show.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) # Assuming 224x224 input
    img = np.float32(img) / 255
    
    visualization = show_cam_on_image(img, heatmap, use_rgb=True)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(visualization)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix using seaborn/matplotlib.
    """
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
