
"""
SHAP explainer για ερμηνεύσιμη βαθιά μάθηση.

Αυτό το module υλοποιεί τη μέθοδο SHAP (SHapley Additive exPlanations)
για την εξήγηση των αποφάσεων του CNN μοντέλου.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Tuple, List, Optional
import shap

from abc import ABC, abstractmethod
from models.cnn_model import create_model


class SHAPExplainer(ABC):
    """
    SHAP explainer για εξήγηση αποφάσεων CNN.
    
    Αυτή η κλάση υλοποιεί τη μέθοδο SHAP που χρησιμοποιεί Shapley values
    για να εξηγήσει τη συμβολή κάθε feature στην πρόβλεψη.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του SHAP explainer.
        
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
        
        # Δημιουργία φακέλου για SHAP αποτελέσματα
        self.shap_dir = os.path.join(output_dir, "shap")
        os.makedirs(self.shap_dir, exist_ok=True)
        
        # Φόρτωση μοντέλου και data loader
        self._load_model()
        self._setup_data_loader()
        self._setup_shap_explainer()
    
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
        
        # Χρήση validation transforms για SHAP
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
    
    def _setup_shap_explainer(self):
        """
        Ρύθμιση του SHAP explainer.
        """
        self.logger.info("Ρύθμιση SHAP explainer...")
        
        # Δημιουργία background dataset για SHAP
        self.background_data = self._create_background_dataset()
        
        # Δημιουργία SHAP explainer
        self.explainer = shap.DeepExplainer(
            self.model,
            self.background_data
        )
        
        self.logger.info("SHAP explainer ρυθμίστηκε επιτυχώς")
    
    def _create_background_dataset(self, num_samples: int = 100) -> torch.Tensor:
        """
        Δημιουργία background dataset για SHAP.
        
        Args:
            num_samples: Αριθμός δειγμάτων για background
            
        Returns:
            torch.Tensor: Background dataset
        """
        self.logger.info(f"Δημιουργία background dataset με {num_samples} δείγματα")
        
        background_samples = []
        
        # Επιλογή τυχαίων δειγμάτων
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        for idx in indices:
            image, _ = self.dataset[idx]
            background_samples.append(image)
        
        background_tensor = torch.stack(background_samples).to(self.device)
        
        self.logger.info(f"Background dataset δημιουργήθηκε: {background_tensor.shape}")
        
        return background_tensor
    
    def explain_sample(self, sample_idx: int = 0):
        """
        Εφαρμογή SHAP σε ένα δείγμα.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            
        Returns:
            dict: Αποτελέσματα εξήγησης
        """
        self.logger.info(f"Εφαρμογή SHAP σε δείγμα {sample_idx}")
        
        # Λήψη δείγματος
        image_tensor, label = self.dataset[sample_idx]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Πρόβλεψη μοντέλου
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Δημιουργία SHAP explanation
        shap_values = self.explainer.shap_values(image_tensor)
        
        # Μετατροπή σε numpy για visualization
        shap_values_np = np.array(shap_values)
        
        # Δημιουργία visualization
        self._create_shap_visualization(
            image_tensor, shap_values_np, predicted_class, confidence, label, sample_idx
        )
        
        # Αποθήκευση αποτελεσμάτων
        filename = f"shap_sample_{sample_idx}_{self.class_names[predicted_class]}.png"
        output_path = os.path.join(self.shap_dir, filename)
        
        self._save_explanation_plot(
            image_tensor, shap_values_np, predicted_class, confidence, label, output_path
        )
        
        self.logger.info(f"SHAP αποθηκεύτηκε: {output_path}")
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'output_path': output_path,
            'shap_values': shap_values_np
        }
    
    def explain_batch(self, batch_size: int = 5):
        """
        Εφαρμογή SHAP σε batch δειγμάτων.
        
        Args:
            batch_size: Μέγεθος batch
        """
        self.logger.info(f"Εφαρμογή SHAP σε batch {batch_size} δειγμάτων")
        
        results = []
        
        for i in range(min(batch_size, len(self.dataset))):
            try:
                result = self.explain_sample(i)
                results.append(result)
                self.logger.info(f"Δείγμα {i}: {result['predicted_class']} -> {result['confidence']:.3f}")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο δείγμα {i}: {e}")
        
        return results
    
    def _create_shap_visualization(self, image_tensor: torch.Tensor, shap_values: np.ndarray,
                                 predicted_class: int, confidence: float, true_label: int, sample_idx: int):
        """
        Δημιουργία SHAP visualization.
        
        Args:
            image_tensor: Είσοδος εικόνα
            shap_values: SHAP values
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            sample_idx: Δείκτης δείγματος
        """
        # Μετατροπή tensor σε εικόνα
        original_image = self._tensor_to_image(image_tensor.squeeze())
        
        # Δημιουργία SHAP heatmap
        shap_heatmap = self._create_shap_heatmap(shap_values, predicted_class)
        
        # Δημιουργία comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Αρχική εικόνα
        axes[0].imshow(original_image)
        axes[0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0].axis('off')
        
        # Βελτιωμένη απεικόνιση SHAP heatmap (absolute values + plasma colormap)
        abs_heatmap = np.abs(shap_heatmap)

        # Υπολογισμός βέλτιστου range
        abs_max = abs_heatmap.max()
        if abs_max < 0.05:   # πολύ μικρές τιμές
            vmin, vmax = 0, abs_max * 2  # λίγο πιο μεγάλο range για να φαίνεται
        else:
            vmin, vmax = 0, abs_max * 1.5

        # Κλιμάκωση heatmap στο [0,1] ώστε να είναι ορατό ακόμη και με μικρά SHAP values
        if abs_heatmap.max() > 0:
            scaled_heatmap = abs_heatmap / abs_heatmap.max()
        else:
            scaled_heatmap = abs_heatmap

        im = axes[1].imshow(scaled_heatmap, cmap='plasma', vmin=0, vmax=1)
        vmin, vmax = 0, 1  # Για το colorbar
        axes[1].set_title(f"SHAP Explanation\nPredicted: {self.class_names[predicted_class]}")
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        overlay = self._overlay_shap_heatmap(original_image, shap_heatmap)
        axes[2].imshow(overlay)
        axes[2].set_title(f"SHAP Overlay\nConfidence: {confidence:.3f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"shap_visualization_sample_{sample_idx}.png"
        output_path = os.path.join(self.shap_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"SHAP visualization αποθηκεύτηκε: {output_path}")
    
    def _create_shap_heatmap(self, shap_values: np.ndarray, target_class: int) -> np.ndarray:
        """
        Δημιουργία SHAP heatmap.
        
        Args:
            shap_values: SHAP values (num_classes, channels, height, width)
            target_class: Τάργκετ κλάση
            
        Returns:
            np.ndarray: SHAP heatmap
        """
        # Λήψη SHAP values για την target class
        class_shap_values = shap_values[target_class]

        # ------------------------------------------------------------
        # 1. Ενοποίηση σχήματος → (H, W)
        #   • Πιθανά σχήματα: (C, H, W)  ή  (1, C, H, W)  ή  (H, W)
        # ------------------------------------------------------------
        if class_shap_values.ndim == 4:               # (1, C, H, W)
            class_shap_values = class_shap_values[0]

        if class_shap_values.ndim == 3:               # (C, H, W)
            # Συνάθροιση απολύτων τιμών κατά μήκος των καναλιών
            heatmap = np.sum(np.abs(class_shap_values), axis=0)
        elif class_shap_values.ndim == 2:             # (H, W)
            heatmap = np.abs(class_shap_values)
        else:
            # Απρόβλεπτο σχήμα → flatten και επαναφορά σε τετράγωνο περίπου
            flat = np.abs(class_shap_values).flatten()
            side = int(np.sqrt(len(flat)))
            heatmap = flat[: side * side].reshape(side, side)

        # ------------------------------------------------------------
        # 2. Κανονικοποίηση για οπτική κλίμακα 0-1
        # ------------------------------------------------------------
        abs_max = heatmap.max()
        if abs_max < 1e-6:
            # Σχεδόν μηδενικά SHAP values → επιστρέφουμε μηδενικό heatmap
            return np.zeros_like(heatmap)

        heatmap /= abs_max  # κλίμακα 0-1

        return heatmap
    
    def _overlay_shap_heatmap(self, original_image: np.ndarray, shap_heatmap: np.ndarray,
                             alpha: float = 0.6) -> np.ndarray:
        """
        Επικάλυψη SHAP heatmap στην αρχική εικόνα.
        
        Args:
            original_image: Αρχική εικόνα (H, W, C)
            shap_heatmap: SHAP heatmap (H, W)
            alpha: Διαφάνεια του heatmap
            
        Returns:
            np.ndarray: Εικόνα με επικάλυψη
        """
        # Μετατροπή heatmap σε RGB
        heatmap_rgb = np.zeros_like(original_image)
        
        # Χρήση red-blue colormap
        positive_mask = shap_heatmap > 0
        negative_mask = shap_heatmap < 0
        
        heatmap_rgb[positive_mask, 0] = shap_heatmap[positive_mask]  # Red για θετικές τιμές
        heatmap_rgb[negative_mask, 2] = -shap_heatmap[negative_mask]  # Blue για αρνητικές τιμές
        
        # Επικάλυψη
        overlay = alpha * heatmap_rgb + (1 - alpha) * original_image
        
        # Clipping σε [0, 1]
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
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
        image = denormalized.cpu().numpy().transpose(1, 2, 0)
        
        # Clipping σε [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def _save_explanation_plot(self, image_tensor: torch.Tensor, shap_values: np.ndarray,
                              predicted_class: int, confidence: float, true_label: int, output_path: str):
        """
        Αποθήκευση plot με SHAP αποτελέσματα.
        
        Args:
            image_tensor: Είσοδος εικόνα
            shap_values: SHAP values
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            output_path: Διαδρομή αποθήκευσης
        """
        original_image = self._tensor_to_image(image_tensor.squeeze())
        shap_heatmap = self._create_shap_heatmap(shap_values, predicted_class)
        overlay = self._overlay_shap_heatmap(original_image, shap_heatmap)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Αρχική εικόνα
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0, 0].axis('off')
        
        # SHAP heatmap
        im = axes[0, 1].imshow(shap_heatmap, cmap='RdBu', 
                               vmin=-np.abs(shap_heatmap).max(), vmax=np.abs(shap_heatmap).max())
        axes[0, 1].set_title(f"SHAP Heatmap\nPredicted: {self.class_names[predicted_class]}")
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Overlay
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(f"SHAP Overlay\nConfidence: {confidence:.3f}")
        axes[1, 0].axis('off')
        
        # SHAP values distribution
        shap_flat = shap_heatmap.flatten()
        axes[1, 1].hist(shap_flat, bins=50, alpha=0.7, color='blue')
        axes[1, 1].set_title("SHAP Values Distribution")
        axes[1, 1].set_xlabel("SHAP Value")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def explain_with_global_context(self, sample_idx: int = 0, num_background: int = 50):
        """
        Εφαρμογή SHAP με global context.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            num_background: Αριθμός background δειγμάτων
            
        Returns:
            dict: Αποτελέσματα εξήγησης
        """
        self.logger.info(f"Εφαρμογή SHAP με global context - δείγμα {sample_idx}")
        
        # Δημιουργία νέου background dataset
        background_data = self._create_background_dataset(num_background)
        
        # Δημιουργία νέου explainer
        explainer = shap.DeepExplainer(self.model, background_data)
        
        # Λήψη δείγματος
        image_tensor, label = self.dataset[sample_idx]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Πρόβλεψη
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # SHAP explanation
        shap_values = explainer.shap_values(image_tensor)
        
        # Δημιουργία global visualization
        self._create_global_shap_visualization(
            image_tensor, shap_values, predicted_class, confidence, label, sample_idx
        )
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'shap_values': np.array(shap_values),
            'num_background': num_background
        }
    
    def _create_global_shap_visualization(self, image_tensor: torch.Tensor, shap_values: list,
                                        predicted_class: int, confidence: float, true_label: int, sample_idx: int):
        """
        Δημιουργία global SHAP visualization.
        
        Args:
            image_tensor: Είσοδος εικόνα
            shap_values: SHAP values για όλες τις κλάσεις
            predicted_class: Προβλεφθείσα κλάση
            confidence: Εμπιστοσύνη πρόβλεψης
            true_label: Πραγματική ετικέτα
            sample_idx: Δείκτης δείγματος
        """
        original_image = self._tensor_to_image(image_tensor.squeeze())
        
        # Δημιουργία heatmaps για όλες τις κλάσεις
        heatmaps = []
        for class_idx in range(len(self.class_names)):
            heatmap = self._create_shap_heatmap(np.array(shap_values), class_idx)
            heatmaps.append(heatmap)
        
        # Δημιουργία comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Αρχική εικόνα
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0, 0].axis('off')
        
        # Heatmaps για κάθε κλάση
        for i, (class_name, heatmap) in enumerate(zip(self.class_names, heatmaps)):
            row, col = (i + 1) // 2, (i + 1) % 2
            im = axes[row, col].imshow(heatmap, cmap='RdBu',
                                      vmin=-np.abs(heatmap).max(), vmax=np.abs(heatmap).max())
            axes[row, col].set_title(f"{class_name.title()}\nConfidence: {confidence:.3f}")
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"shap_global_sample_{sample_idx}.png"
        output_path = os.path.join(self.shap_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Global SHAP visualization αποθηκεύτηκε: {output_path}")
    
    def compare_with_other_methods(self, sample_idx: int = 0):
        """
        Σύγκριση SHAP με άλλες μεθόδους (simulation).
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        self.logger.info(f"Σύγκριση SHAP με άλλες μεθόδους - δείγμα {sample_idx}")
        
        # Εφαρμογή SHAP
        shap_result = self.explain_sample(sample_idx)
        
        # Simulation άλλων μεθόδων (για demonstration)
        # Στην πραγματικότητα θα χρησιμοποιούσαμε τα πραγματικά explainers
        
        comparison_results = {
            'shap': {
                'method': 'SHAP',
                'predicted_class': shap_result['predicted_class'],
                'confidence': shap_result['confidence'],
                'explanation_quality': 'High',
                'computation_time': 'Medium'
            },
            'gradcam': {
                'method': 'Grad-CAM',
                'predicted_class': shap_result['predicted_class'],
                'confidence': shap_result['confidence'],
                'explanation_quality': 'Medium',
                'computation_time': 'Fast'
            },
            'lime': {
                'method': 'LIME',
                'predicted_class': shap_result['predicted_class'],
                'confidence': shap_result['confidence'],
                'explanation_quality': 'High',
                'computation_time': 'Slow'
            }
        }
        
        # Δημιουργία comparison plot
        self._create_method_comparison_plot(comparison_results, sample_idx)
        
        return comparison_results
    
    def _create_method_comparison_plot(self, comparison_results: dict, sample_idx: int):
        """
        Δημιουργία plot για σύγκριση μεθόδων.
        
        Args:
            comparison_results: Αποτελέσματα σύγκρισης
            sample_idx: Δείκτης δείγματος
        """
        methods = list(comparison_results.keys())
        confidences = [comparison_results[method]['confidence'] for method in methods]
        qualities = [comparison_results[method]['explanation_quality'] for method in methods]
        
        # Mapping quality σε αριθμητικές τιμές
        quality_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        quality_values = [quality_mapping[q] for q in qualities]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence comparison
        bars1 = ax1.bar(methods, confidences, color=['red', 'blue', 'green'])
        ax1.set_title('Confidence Comparison')
        ax1.set_ylabel('Confidence')
        ax1.set_ylim(0, 1)
        
        # Quality comparison
        bars2 = ax2.bar(methods, quality_values, color=['orange', 'purple', 'brown'])
        ax2.set_title('Explanation Quality Comparison')
        ax2.set_ylabel('Quality (1=Low, 2=Medium, 3=High)')
        ax2.set_ylim(0, 4)
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"method_comparison_sample_{sample_idx}.png"
        output_path = os.path.join(self.shap_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Method comparison αποθηκεύτηκε: {output_path}") 