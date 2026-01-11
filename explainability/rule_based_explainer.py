
"""
Rule-based explainer για ερμηνεύσιμη βαθιά μάθηση.

Αυτό το module υλοποιεί ένα rule-based system που εξηγεί τις αποφάσεις
του CNN μοντέλου με ρητούς κανόνες για γάτες και σκύλους.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from abc import ABC, abstractmethod
from models.cnn_model import create_model


class RuleBasedExplainer(ABC):
    """
    Rule-based explainer για εξήγηση αποφάσεων CNN.
    
    Αυτή η κλάση υλοποιεί ένα σύστημα κανόνων που εξηγεί τις αποφάσεις
    του μοντέλου με ρητούς κανόνες για ταξινόμηση γάτας/σκύλου.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του rule-based explainer.
        
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
        
        # Δημιουργία φακέλου για rule-based αποτελέσματα
        self.rules_dir = os.path.join(output_dir, "rule_based")
        os.makedirs(self.rules_dir, exist_ok=True)
        
        # Φόρτωση μοντέλου και data loader
        self._load_model()
        self._setup_data_loader()
        self._setup_rules()
    
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
        
        # Χρήση validation transforms
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
    
    def _setup_rules(self):
        """
        Ρύθμιση των κανόνων για εξήγηση.
        """
        self.logger.info("Ρύθμιση κανόνων εξήγησης...")
        
        # Κανόνες για γάτες
        self.cat_rules = {
            "pointed_ears": {
                "description": "Μυτερά αυτιά",
                "weight": 0.3,
                "explanation": "Η γάτα έχει μυτερά, τριγωνικά αυτιά"
            },
            "slender_body": {
                "description": "Λεπτό σώμα",
                "weight": 0.2,
                "explanation": "Η γάτα έχει λεπτότερο και πιο κομψό σώμα"
            },
            "long_tail": {
                "description": "Μακριά ουρά",
                "weight": 0.25,
                "explanation": "Η γάτα έχει μακριά και λεπτή ουρά"
            },
            "pointed_face": {
                "description": "Μυτερό πρόσωπο",
                "weight": 0.15,
                "explanation": "Η γάτα έχει μυτερότερο και πιο λεπτό πρόσωπο"
            },
            "vertical_pupils": {
                "description": "Κάθετες κόρες",
                "weight": 0.1,
                "explanation": "Η γάτα έχει κάθετες κόρες στα μάτια"
            }
        }
        
        # Κανόνες για σκύλους
        self.dog_rules = {
            "floppy_ears": {
                "description": "Κρεμαστά αυτιά",
                "weight": 0.25,
                "explanation": "Ο σκύλος έχει κρεμαστά αυτιά"
            },
            "stocky_body": {
                "description": "Παχύ σώμα",
                "weight": 0.3,
                "explanation": "Ο σκύλος έχει πιο παχύ και δυνατό σώμα"
            },
            "short_tail": {
                "description": "Κοντή ουρά",
                "weight": 0.2,
                "explanation": "Ο σκύλος έχει κοντύτερη και πιο παχιά ουρά"
            },
            "broad_face": {
                "description": "Πλατύ πρόσωπο",
                "weight": 0.15,
                "explanation": "Ο σκύλος έχει πλατύτερο και πιο στρογγυλό πρόσωπο"
            },
            "round_pupils": {
                "description": "Στρογγυλές κόρες",
                "weight": 0.1,
                "explanation": "Ο σκύλος έχει στρογγυλές κόρες στα μάτια"
            }
        }
        
        self.logger.info("Κανόνες εξήγησης ρυθμίστηκαν επιτυχώς")
    
    def _extract_features(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Εξαγωγή χαρακτηριστικών από την εικόνα.
        
        Args:
            image: Είσοδος εικόνα (1, C, H, W)
            
        Returns:
            Dict[str, float]: Λεξικό με τα εξαγόμενα χαρακτηριστικά
        """
        # Μετατροπή σε numpy για OpenCV
        image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Μετατροπή σε grayscale για ανάλυση
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Ανάλυση αυτιών
        features['pointed_ears'] = self._detect_pointed_ears(gray)
        features['floppy_ears'] = self._detect_floppy_ears(gray)
        
        # Ανάλυση σώματος
        features['slender_body'] = self._detect_slender_body(gray)
        features['stocky_body'] = self._detect_stocky_body(gray)
        
        # Ανάλυση ουράς
        features['long_tail'] = self._detect_long_tail(gray)
        features['short_tail'] = self._detect_short_tail(gray)
        
        # Ανάλυση προσώπου
        features['pointed_face'] = self._detect_pointed_face(gray)
        features['broad_face'] = self._detect_broad_face(gray)
        
        # Ανάλυση ματιών
        features['vertical_pupils'] = self._detect_vertical_pupils(gray)
        features['round_pupils'] = self._detect_round_pupils(gray)
        
        return features
    
    def _detect_pointed_ears(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση μυτερών αυτιών.
        """
        # Απλή υλοποίηση - μπορεί να επεκταθεί
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            vertical_lines = sum(1 for line in lines if abs(line[0][1] - np.pi/2) < 0.3)
            return min(vertical_lines / 10.0, 1.0)
        return 0.0
    
    def _detect_floppy_ears(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση κρεμαστών αυτιών.
        """
        # Απλή υλοποίηση
        return 1.0 - self._detect_pointed_ears(gray)
    
    def _detect_slender_body(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση λεπτού σώματος.
        """
        # Υπολογισμός aspect ratio
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = h / w if w > 0 else 1
            return min(aspect_ratio / 2.0, 1.0)
        return 0.5
    
    def _detect_stocky_body(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση παχύ σώματος.
        """
        return 1.0 - self._detect_slender_body(gray)
    
    def _detect_long_tail(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση μακριάς ουράς.
        """
        # Απλή υλοποίηση
        return 0.6
    
    def _detect_short_tail(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση κοντής ουράς.
        """
        return 1.0 - self._detect_long_tail(gray)
    
    def _detect_pointed_face(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση μυτερού προσώπου.
        """
        # Απλή υλοποίηση
        return 0.7
    
    def _detect_broad_face(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση πλατιού προσώπου.
        """
        return 1.0 - self._detect_pointed_face(gray)
    
    def _detect_vertical_pupils(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση κάθετων κορών.
        """
        # Απλή υλοποίηση
        return 0.8
    
    def _detect_round_pupils(self, gray: np.ndarray) -> float:
        """
        Ανίχνευση στρογγυλών κορών.
        """
        return 1.0 - self._detect_vertical_pupils(gray)
    
    def apply_rules(self, features: Dict[str, float]) -> Tuple[str, float, List[str]]:
        """
        Εφαρμογή κανόνων για εξήγηση.
        
        Args:
            features: Εξαγόμενα χαρακτηριστικά
            
        Returns:
            Tuple[str, float, List[str]]: (κλάση, εμπιστοσύνη, εξηγήσεις)
        """
        cat_score = 0.0
        dog_score = 0.0
        cat_explanations = []
        dog_explanations = []
        
        # Εφαρμογή κανόνων για γάτες
        for rule_name, rule_info in self.cat_rules.items():
            if rule_name in features:
                score = features[rule_name] * rule_info['weight']
                cat_score += score
                if features[rule_name] > 0.5:
                    cat_explanations.append(f"✓ {rule_info['explanation']}")
        
        # Εφαρμογή κανόνων για σκύλους
        for rule_name, rule_info in self.dog_rules.items():
            if rule_name in features:
                score = features[rule_name] * rule_info['weight']
                dog_score += score
                if features[rule_name] > 0.5:
                    dog_explanations.append(f"✓ {rule_info['explanation']}")
        
        # Κανονικοποίηση scores
        cat_score = min(cat_score, 1.0)
        dog_score = min(dog_score, 1.0)
        
        # Επιλογή κλάσης
        if cat_score > dog_score:
            predicted_class = "cat"
            confidence = cat_score
            explanations = cat_explanations
        else:
            predicted_class = "dog"
            confidence = dog_score
            explanations = dog_explanations
        
        return predicted_class, confidence, explanations
    
    def explain_sample(self, sample_idx: int = 0):
        """
        Εφαρμογή rule-based εξήγησης σε ένα δείγμα.
        
        Args:
            sample_idx: Δείκτης του δείγματος
            
        Returns:
            dict: Αποτελέσματα εξήγησης
        """
        self.logger.info(f"Εφαρμογή rule-based εξήγησης σε δείγμα {sample_idx}")
        
        # Λήψη δείγματος
        image_tensor, label = self.dataset[sample_idx]
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Πρόβλεψη μοντέλου
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            model_predicted_class = torch.argmax(output, dim=1).item()
            model_confidence = probabilities[0, model_predicted_class].item()
        
        # Εξαγωγή χαρακτηριστικών
        features = self._extract_features(image_tensor)
        
        # Εφαρμογή κανόνων
        rule_predicted_class, rule_confidence, explanations = self.apply_rules(features)
        
        # Δημιουργία visualization
        self._create_rule_visualization(
            image_tensor, features, rule_predicted_class, rule_confidence, 
            model_predicted_class, model_confidence, label, explanations, sample_idx
        )
        
        # Αποθήκευση αποτελεσμάτων
        filename = f"rule_based_sample_{sample_idx}_{rule_predicted_class}.png"
        output_path = os.path.join(self.rules_dir, filename)
        
        self._save_explanation_plot(
            image_tensor, features, rule_predicted_class, rule_confidence,
            model_predicted_class, model_confidence, label, explanations, output_path
        )
        
        self.logger.info(f"Rule-based εξήγηση αποθηκεύτηκε: {output_path}")
        
        return {
            'sample_idx': sample_idx,
            'true_label': label,
            'model_predicted_class': model_predicted_class,
            'model_confidence': model_confidence,
            'rule_predicted_class': rule_predicted_class,
            'rule_confidence': rule_confidence,
            'explanations': explanations,
            'features': features,
            'output_path': output_path
        }
    
    def _create_rule_visualization(self, image_tensor: torch.Tensor, features: Dict[str, float],
                                  rule_predicted_class: str, rule_confidence: float,
                                  model_predicted_class: int, model_confidence: float,
                                  true_label: int, explanations: List[str], sample_idx: int):
        """
        Δημιουργία visualization για rule-based εξήγηση.
        """
        # Μετατροπή εικόνας
        image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Δημιουργία figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Αρχική εικόνα
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title(f"Αρχική Εικόνα\nTrue: {self.class_names[true_label]}")
        axes[0, 0].axis('off')
        
        # Χαρακτηριστικά
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        axes[0, 1].barh(feature_names, feature_values, color='skyblue', alpha=0.7)
        axes[0, 1].set_title("Εξαγόμενα Χαρακτηριστικά")
        axes[0, 1].set_xlabel("Τιμή")
        
        # Σύγκριση προβλέψεων
        classes = ['Cat', 'Dog']
        model_probs = [0, 0]
        model_probs[model_predicted_class] = model_confidence
        
        rule_probs = [0, 0]
        if rule_predicted_class == "cat":
            rule_probs[0] = rule_confidence
        else:
            rule_probs[1] = rule_confidence
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, model_probs, width, label='Model', alpha=0.7)
        axes[1, 0].bar(x + width/2, rule_probs, width, label='Rules', alpha=0.7)
        axes[1, 0].set_title("Σύγκριση Προβλέψεων")
        axes[1, 0].set_ylabel("Εμπιστοσύνη")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes)
        axes[1, 0].legend()
        
        # Εξηγήσεις
        explanation_text = f"Rule-based Εξήγηση:\n\n"
        explanation_text += f"Πρόβλεψη: {rule_predicted_class.title()}\n"
        explanation_text += f"Εμπιστοσύνη: {rule_confidence:.3f}\n\n"
        explanation_text += "Εφαρμοσμένοι Κανόνες:\n"
        
        for explanation in explanations:
            explanation_text += f"• {explanation}\n"
        
        axes[1, 1].text(0.1, 0.9, explanation_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title("Εξηγήσεις Κανόνων")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Αποθήκευση
        filename = f"rule_based_sample_{sample_idx}_{rule_predicted_class}.png"
        output_path = os.path.join(self.rules_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_explanation_plot(self, image_tensor: torch.Tensor, features: Dict[str, float],
                               rule_predicted_class: str, rule_confidence: float,
                               model_predicted_class: int, model_confidence: float,
                               true_label: int, explanations: List[str], output_path: str):
        """
        Αποθήκευση rule-based εξήγησης.
        """
        self._create_rule_visualization(
            image_tensor, features, rule_predicted_class, rule_confidence,
            model_predicted_class, model_confidence, true_label, explanations, 0
        )
    
    def explain_batch(self, batch_size: int = 5):
        """
        Εφαρμογή rule-based εξήγησης σε batch δειγμάτων.
        
        Args:
            batch_size: Μέγεθος batch
        """
        self.logger.info(f"Εφαρμογή rule-based εξήγησης σε batch {batch_size} δειγμάτων")
        
        results = []
        
        for i in range(min(batch_size, len(self.dataset))):
            try:
                result = self.explain_sample(i)
                results.append(result)
                self.logger.info(f"Δείγμα {i}: {result['rule_predicted_class']} -> {result['rule_confidence']:.3f}")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο δείγμα {i}: {e}")
        
        return results
    
    def compare_with_model(self, sample_idx: int = 0):
        """
        Σύγκριση rule-based με model predictions.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        result = self.explain_sample(sample_idx)
        
        agreement = result['model_predicted_class'] == (0 if result['rule_predicted_class'] == 'cat' else 1)
        
        self.logger.info(f"Σύγκριση για δείγμα {sample_idx}:")
        self.logger.info(f"  Model: {self.class_names[result['model_predicted_class']]} ({result['model_confidence']:.3f})")
        self.logger.info(f"  Rules: {result['rule_predicted_class']} ({result['rule_confidence']:.3f})")
        self.logger.info(f"  Συμφωνία: {'Ναι' if agreement else 'Όχι'}")
        
        return result 