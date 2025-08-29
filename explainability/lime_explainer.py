
"""
LIME explainer για ερμηνεύσιμη βαθιά μάθηση.

Αυτό το module υλοποιεί τη μέθοδο LIME (Local Interpretable Model-agnostic Explanations)
για την εξήγηση των αποφάσεων του CNN μοντέλου.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Tuple, List
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from abc import ABC, abstractmethod
from models.cnn_model import create_model


class LIMEExplainer(ABC):
    """
    LIME explainer για εξήγηση αποφάσεων CNN.
    
    Αυτή η κλάση υλοποιεί τη μέθοδο LIME που δημιουργεί τοπικές
    ερμηνείες χρησιμοποιώντας γραμμικά μοντέλα σε perturbed δείγματα.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του LIME explainer.
        
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
        
        # Δημιουργία φακέλου για LIME αποτελέσματα
        self.lime_dir = os.path.join(output_dir, "lime")
        os.makedirs(self.lime_dir, exist_ok=True)
        
        # Φόρτωση μοντέλου και data loader
        self._load_model()
        self._setup_data_loader()
        self._setup_lime_explainer()
    
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
        
        # Χρήση validation transforms για LIME
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
    
    def _setup_lime_explainer(self):
        """
        Ρύθμιση του LIME explainer.
        """
        self.logger.info("Ρύθμιση LIME explainer...")
        
        # Δημιουργία LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
        
        # Segmentation algorithm για superpixels
        self.segmenter = SegmentationAlgorithm(
            'quickshift',
            kernel_size=4,
            max_dist=200,
            ratio=0.2
        )
        
        self.logger.info("LIME explainer ρυθμίστηκε επιτυχώς")
    
    def _model_predict(self, images: np.ndarray) -> np.ndarray:
        """
        Πρόβλεψη μοντέλου για LIME.
        
        Args:
            images: Εικόνες σε numpy format (N, H, W, C)
            
        Returns:
            np.ndarray: Προβλέψεις (N, num_classes)
        """
        predictions = []
        
        for image in images:
            # Μετατροπή σε tensor
            image_tensor = self._numpy_to_tensor(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Πρόβλεψη
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        return np.array(predictions).squeeze()
    
    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Μετατροπή numpy εικόνας σε PyTorch tensor.
        
        Args:
            image: Numpy εικόνα (H, W, C) σε [0, 1]
            
        Returns:
            torch.Tensor: Normalized tensor (C, H, W)
        """
        # Μετατροπή σε PIL Image
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Εφαρμογή transforms
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image_pil)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Μετατροπή PyTorch tensor σε numpy εικόνα.
        
        Args:
            tensor: Normalized tensor (C, H, W)
            
        Returns:
            np.ndarray: Numpy εικόνα (H, W, C) σε [0, 1]
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
    
    def explain_sample(self, sample_idx: int = 0):
        """
        Εφαρμογή LIME σε ένα δείγμα.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            
        Returns:
            dict: Αποτελέσματα εξήγησης
        """
        self.logger.info(f"Εφαρμογή LIME σε δείγμα {sample_idx}")
        
        # Λήψη δείγματος
        image_tensor, label = self.dataset[sample_idx]
        image_numpy = self._tensor_to_numpy(image_tensor)
        
        # Πρόβλεψη μοντέλου
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Δημιουργία LIME explanation
        explanation = self.explainer.explain_instance(
            image_numpy,
            self._model_predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000,
            segmentation_fn=self.segmenter
        )
        
        # Λήψη superpixels και βαρών
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        
        # Δημιουργία visualization
        self._create_lime_visualization(
            image_numpy, temp, mask, predicted_class, confidence, label, sample_idx
        )
        
        # Αποθήκευση αποτελεσμάτων
        filename = f"lime_sample_{sample_idx}_{self.class_names[predicted_class]}.png"
        output_path = os.path.join(self.lime_dir, filename)
        
        self._save_explanation_plot(
            image_numpy, temp, mask, predicted_class, confidence, label, output_path
        )
        
        self.logger.info(f"LIME αποθηκεύτηκε: {output_path}")
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'output_path': output_path,
            'explanation': explanation
        }
    
    def explain_batch(self, batch_size: int = 5):
        """
        Εφαρμογή LIME σε batch δειγμάτων.
        
        Args:
            batch_size: Μέγεθος batch
        """
        self.logger.info(f"Εφαρμογή LIME σε batch {batch_size} δειγμάτων")
        
        results = []
        
        for i in range(min(batch_size, len(self.dataset))):
            try:
                result = self.explain_sample(i)
                results.append(result)
                self.logger.info(f"Δείγμα {i}: {result['predicted_class']} -> {result['confidence']:.3f}")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο δείγμα {i}: {e}")
        
        return results
    
    def _create_lime_visualization(self, original_image: np.ndarray, temp: np.ndarray,
                                 mask: np.ndarray, predicted_class: int, confidence: float,
                                 true_label: int, sample_idx: int):
        """
        Δημιουργία LIME visualization.
        
        Args:
            original_image: Αρχική εικόνα
            temp: LIME explanation image
            mask: Superpixel mask
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            sample_idx: Δείκτης δείγματος
        """
        # Δημιουργία comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Αρχική εικόνα
        axes[0].imshow(original_image)
        axes[0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0].axis('off')
        
        # LIME explanation
        axes[1].imshow(temp)
        axes[1].set_title(f"LIME Explanation\nPredicted: {self.class_names[predicted_class]}")
        axes[1].axis('off')
        
        # Superpixel mask
        axes[2].imshow(mask, cmap='Reds', alpha=0.7)
        axes[2].imshow(original_image, alpha=0.3)
        axes[2].set_title(f"Superpixel Mask\nConfidence: {confidence:.3f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"lime_visualization_sample_{sample_idx}.png"
        output_path = os.path.join(self.lime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"LIME visualization αποθηκεύτηκε: {output_path}")
    
    def _save_explanation_plot(self, original_image: np.ndarray, temp: np.ndarray,
                              mask: np.ndarray, predicted_class: int, confidence: float,
                              true_label: int, output_path: str):
        """
        Αποθήκευση plot με LIME αποτελέσματα.
        
        Args:
            original_image: Αρχική εικόνα
            temp: LIME explanation image
            mask: Superpixel mask
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            output_path: Διαδρομή αποθήκευσης
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Αρχική εικόνα
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0, 0].axis('off')
        
        # LIME explanation
        axes[0, 1].imshow(temp)
        axes[0, 1].set_title(f"LIME Explanation\nPredicted: {self.class_names[predicted_class]}")
        axes[0, 1].axis('off')
        
        # Superpixel mask
        axes[1, 0].imshow(mask, cmap='Reds', alpha=0.7)
        axes[1, 0].imshow(original_image, alpha=0.3)
        axes[1, 0].set_title(f"Superpixel Mask\nConfidence: {confidence:.3f}")
        axes[1, 0].axis('off')
        
        # Feature importance (αν είναι διαθέσιμο)
        try:
            # Δημιουργία feature importance plot
            feature_importance = self._get_feature_importance(mask)
            axes[1, 1].bar(range(len(feature_importance)), feature_importance)
            axes[1, 1].set_title("Feature Importance")
            axes[1, 1].set_xlabel("Superpixel Index")
            axes[1, 1].set_ylabel("Importance")
        except:
            axes[1, 1].text(0.5, 0.5, "Feature Importance\nNot Available", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Feature Importance")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_feature_importance(self, mask: np.ndarray) -> List[float]:
        """
        Υπολογισμός feature importance από το superpixel mask.
        
        Args:
            mask: Superpixel mask
            
        Returns:
            List[float]: Λίστα με τα βάρη των features
        """
        # Απλή υλοποίηση - μπορεί να επεκταθεί
        unique_segments = np.unique(mask)
        importance = []
        
        for segment in unique_segments:
            if segment > 0:  # Αγνοούμε το background
                segment_area = np.sum(mask == segment)
                importance.append(segment_area / mask.size)
        
        return importance
    
    def explain_with_confidence_threshold(self, sample_idx: int = 0, threshold: float = 0.8):
        """
        Εφαρμογή LIME μόνο για προβλέψεις με υψηλή εμπιστοσύνη.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            threshold: Όριο εμπιστοσύνης
            
        Returns:
            dict: Αποτελέσματα εξήγησης ή None αν η εμπιστοσύνη είναι χαμηλή
        """
        # Λήψη δείγματος
        image_tensor, label = self.dataset[sample_idx]
        
        # Πρόβλεψη μοντέλου
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        if confidence < threshold:
            self.logger.info(f"Εμπιστοσύνη {confidence:.3f} κάτω από το όριο {threshold}")
            return None
        
        return self.explain_sample(sample_idx)
    
    def compare_explanations(self, sample_idx: int = 0):
        """
        Σύγκριση LIME explanations για διαφορετικές παραμέτρους.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        self.logger.info(f"Σύγκριση LIME explanations - δείγμα {sample_idx}")
        
        # Διαφορετικές παραμέτρους
        num_samples_list = [500, 1000, 2000]
        num_features_list = [5, 10, 15]
        
        results = {}
        
        for num_samples in num_samples_list:
            for num_features in num_features_list:
                try:
                    # Λήψη δείγματος
                    image_tensor, label = self.dataset[sample_idx]
                    image_numpy = self._tensor_to_numpy(image_tensor)
                    
                    # Πρόβλεψη
                    with torch.no_grad():
                        output = self.model(image_tensor.unsqueeze(0).to(self.device))
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(output, dim=1).item()
                        confidence = probabilities[0, predicted_class].item()
                    
                    # LIME explanation με διαφορετικές παραμέτρους
                    explanation = self.explainer.explain_instance(
                        image_numpy,
                        self._model_predict,
                        top_labels=2,
                        hide_color=0,
                        num_samples=num_samples,
                        segmentation_fn=self.segmenter
                    )
                    
                    temp, mask = explanation.get_image_and_mask(
                        predicted_class,
                        positive_only=False,
                        num_features=num_features,
                        hide_rest=False
                    )
                    
                    # Αποθήκευση αποτελέσματος
                    key = f"samples_{num_samples}_features_{num_features}"
                    results[key] = {
                        'temp': temp,
                        'mask': mask,
                        'confidence': confidence,
                        'predicted_class': predicted_class
                    }
                    
                except Exception as e:
                    self.logger.error(f"Σφάλμα για {key}: {e}")
        
        # Δημιουργία comparison plot
        self._create_comparison_plot(results, sample_idx)
        
        return results
    
    def _create_comparison_plot(self, results: dict, sample_idx: int):
        """
        Δημιουργία plot για σύγκριση διαφορετικών παραμέτρων LIME.
        
        Args:
            results: Αποτελέσματα με διαφορετικές παραμέτρους
            sample_idx: Δείκτης δείγματος
        """
        n_results = len(results)
        if n_results == 0:
            return
        
        # Δημιουργία subplot grid
        cols = 3
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (key, result) in enumerate(results.items()):
            row, col = i // cols, i % cols
            
            axes[row, col].imshow(result['temp'])
            axes[row, col].set_title(f"{key}\nConfidence: {result['confidence']:.3f}")
            axes[row, col].axis('off')
        
        # Απόκρυψη κενών subplots
        for i in range(n_results, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"lime_comparison_sample_{sample_idx}.png"
        output_path = os.path.join(self.lime_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"LIME comparison αποθηκεύτηκε: {output_path}") 