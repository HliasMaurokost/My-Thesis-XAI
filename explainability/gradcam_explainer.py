
"""
Grad-CAM explainer για ερμηνεύσιμη βαθιά μάθηση.

Αυτό το module υλοποιεί τη μέθοδο Grad-CAM (Gradient-weighted Class Activation Mapping)
για την εξήγηση των αποφάσεων του CNN μοντέλου.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import logging
from typing import Tuple, Optional

from abc import ABC, abstractmethod
from models.cnn_model import create_model
from utils.data_utils import create_data_loaders


class GradCAMExplainer(ABC):
    """
    Grad-CAM explainer για εξήγηση αποφάσεων CNN.
    
    Αυτή η κλάση υλοποιεί τη μέθοδο Grad-CAM που συνδυάζει gradients
    και activations για να δημιουργήσει heatmaps που δείχνουν τις
    περιοχές της εικόνας που επηρεάζουν περισσότερο την πρόβλεψη.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του Grad-CAM explainer.
        
        Args:
            model_path: Διαδρομή προς το μοντέλο
            dataset_path: Διαδρομή προς το dataset
            output_dir: Φάκελος εξόδου
            device: Συσκευή για εκτέλεση
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Δημιουργία φακέλου για Grad-CAM αποτελέσματα
        self.gradcam_dir = os.path.join(output_dir, "gradcam")
        os.makedirs(self.gradcam_dir, exist_ok=True)
        
        # Φόρτωση μοντέλου και data loader
        self._load_model()
        self._setup_data_loader()
    
    def _load_model(self):
        """
        Φόρτωση του εκπαιδευμένου μοντέλου.
        """
        self.logger.info(f"Φόρτωση μοντέλου από: {self.model_path}")
        
        # Δημιουργία μοντέλου
        self.model = create_model(
            model_type="resnet18",
            num_classes=2,
            pretrained=False
        )
        
        # Φόρτωση weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Μεταφορά στο device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Λήψη class names
        self.class_names = checkpoint.get('class_names', ['cat', 'dog'])
        
        self.logger.info("Μοντέλο φορτώθηκε επιτυχώς")
    
    def _setup_data_loader(self):
        """
        Ρύθμιση του data loader για εξήγηση.
        """
        self.logger.info("Ρύθμιση data loader...")
        
        # Χρήση validation transforms για Grad-CAM
        from torchvision import transforms
        
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Φόρτωση dataset
        from torchvision import datasets
        self.dataset = datasets.ImageFolder(self.dataset_path, transform=val_transforms)
        
        self.logger.info(f"Dataset φορτώθηκε: {len(self.dataset)} εικόνες")
    
    def generate_gradcam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Δημιουργία Grad-CAM heatmap για μια εικόνα.
        
        Args:
            image: Είσοδος εικόνα (1, C, H, W)
            target_class: Τάργκετ κλάση για την οποία δημιουργείται το heatmap
            
        Returns:
            np.ndarray: Grad-CAM heatmap
        """
        # Forward pass
        output = self.model(image)
        
        # Αν το μοντέλο έχει Grad-CAM υποστήριξη
        if hasattr(self.model, 'get_gradcam_weights'):
            # Χρήση built-in Grad-CAM
            weights = self.model.get_gradcam_weights(target_class)
            activations = self.model.get_activations()
            
            # Δημιουργία heatmap
            heatmap = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
            heatmap = F.relu(heatmap)  # Εφαρμογή ReLU
            
        else:
            # Manual Grad-CAM implementation
            heatmap = self._manual_gradcam(image, target_class)
        
        # Resize heatmap στο μέγεθος της εικόνας
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # Μετατροπή σε numpy
        heatmap = heatmap.cpu().numpy()
        
        # Normalization
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _manual_gradcam(self, image: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Manual υλοποίηση Grad-CAM για μοντέλα χωρίς built-in υποστήριξη.
        
        Args:
            image: Είσοδος εικόνα
            target_class: Τάργκετ κλάση
            
        Returns:
            torch.Tensor: Grad-CAM heatmap
        """
        # Εγγραφή gradients
        image.requires_grad_(True)
        
        # Forward pass
        output = self.model(image)
        
        # Backward pass για target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Λήψη gradients
        gradients = image.grad
        
        # Λήψη activations του τελευταίου convolutional layer
        # Χρήση hooks για την εξαγωγή activations
        activations = None
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output
        
        # Εγγραφή hook στο τελευταίο convolutional layer
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(save_activation)
                hooks.append(hook)
        
        # Forward pass για λήψη activations
        with torch.no_grad():
            self.model(image)
        
        # Αφαίρεση hooks
        for hook in hooks:
            hook.remove()
        
        if activations is None:
            raise ValueError("Δεν ήταν δυνατή η λήψη activations")
        
        # Υπολογισμός βαρών
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Δημιουργία heatmap
        heatmap = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
        heatmap = F.relu(heatmap)
        
        return heatmap
    
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.6) -> np.ndarray:
        """
        Επικάλυψη heatmap στην αρχική εικόνα.
        
        Args:
            original_image: Αρχική εικόνα (H, W, C)
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Διαφάνεια του heatmap
            
        Returns:
            np.ndarray: Εικόνα με επικάλυψη
        """
        # Μετατροπή heatmap σε RGB
        heatmap_rgb = np.zeros_like(original_image)
        heatmap_rgb[:, :, 0] = heatmap  # Red channel
        
        # Επικάλυψη
        overlay = alpha * heatmap_rgb + (1 - alpha) * original_image
        
        # Clipping σε [0, 1]
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def explain_sample(self, sample_idx: int = 0):
        """
        Εφαρμογή Grad-CAM σε ένα δείγμα.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            
        Returns:
            dict: Αποτελέσματα εξήγησης
        """
        self.logger.info(f"Εφαρμογή Grad-CAM σε δείγμα {sample_idx}")
        
        # Λήψη δείγματος
        image, label = self.dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(self.device)
        
        # Πρόβλεψη μοντέλου
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Δημιουργία Grad-CAM για την προβλεφθείσα κλάση
        heatmap = self.generate_gradcam(image_tensor, predicted_class)
        
        # Μετατροπή εικόνας για visualization
        original_image = self._tensor_to_image(image)
        
        # Δημιουργία overlay
        overlay = self.overlay_heatmap(original_image, heatmap)
        
        # Αποθήκευση αποτελεσμάτων
        filename = f"gradcam_sample_{sample_idx}_{self.class_names[predicted_class]}.png"
        output_path = os.path.join(self.gradcam_dir, filename)
        
        self._save_explanation_plot(
            original_image, heatmap, overlay, 
            predicted_class, confidence, label, output_path
        )
        
        self.logger.info(f"Grad-CAM αποθηκεύτηκε: {output_path}")
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'output_path': output_path
        }
    
    def explain_batch(self, batch_size: int = 5):
        """
        Εφαρμογή Grad-CAM σε batch δειγμάτων.
        
        Args:
            batch_size: Μέγεθος batch
        """
        self.logger.info(f"Εφαρμογή Grad-CAM σε batch {batch_size} δειγμάτων")
        
        results = []
        
        for i in range(min(batch_size, len(self.dataset))):
            try:
                result = self.explain_sample(i)
                results.append(result)
                self.logger.info(f"Δείγμα {i}: {result['predicted_class']} -> {result['confidence']:.3f}")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο δείγμα {i}: {e}")
        
        return results
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Μετατροπή tensor σε εικόνα για visualization.
        
        Args:
            tensor: PyTorch tensor (C, H, W)
            
        Returns:
            np.ndarray: Εικόνα (H, W, C)
        """
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denormalized = tensor * std + mean
        
        # Μετατροπή σε numpy και transpose
        image = denormalized.numpy().transpose(1, 2, 0)
        
        # Clipping σε [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def _save_explanation_plot(self, original_image: np.ndarray, heatmap: np.ndarray,
                              overlay: np.ndarray, predicted_class: int, confidence: float,
                              true_label: int, output_path: str):
        """
        Αποθήκευση plot με Grad-CAM αποτελέσματα.
        
        Args:
            original_image: Αρχική εικόνα
            heatmap: Grad-CAM heatmap
            overlay: Εικόνα με επικάλυψη
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            output_path: Διαδρομή αποθήκευσης
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Αρχική εικόνα
        axes[0].imshow(original_image)
        axes[0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f"Grad-CAM Heatmap\nPredicted: {self.class_names[predicted_class]}")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nConfidence: {confidence:.3f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_classes(self, sample_idx: int = 0):
        """
        Σύγκριση Grad-CAM για όλες τις κλάσεις.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        self.logger.info(f"Σύγκριση Grad-CAM για όλες τις κλάσεις - δείγμα {sample_idx}")
        
        # Λήψη δείγματος
        image, label = self.dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(self.device)
        
        # Δημιουργία heatmaps για όλες τις κλάσεις
        heatmaps = []
        confidences = []
        
        for class_idx in range(len(self.class_names)):
            heatmap = self.generate_gradcam(image_tensor, class_idx)
            heatmaps.append(heatmap)
            
            # Πρόβλεψη για την κλάση
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities[0, class_idx].item()
                confidences.append(confidence)
        
        # Δημιουργία comparison plot
        original_image = self._tensor_to_image(image)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Αρχική εικόνα
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[label]}")
        axes[0, 0].axis('off')
        
        # Heatmaps για κάθε κλάση
        for i, (class_name, heatmap, confidence) in enumerate(zip(self.class_names, heatmaps, confidences)):
            row, col = (i + 1) // 2, (i + 1) % 2
            axes[row, col].imshow(heatmap, cmap='jet')
            axes[row, col].set_title(f"{class_name.title()}\nConfidence: {confidence:.3f}")
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"gradcam_comparison_sample_{sample_idx}.png"
        output_path = os.path.join(self.gradcam_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Grad-CAM comparison αποθηκεύτηκε: {output_path}")
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'confidences': confidences,
            'output_path': output_path
        } 