
"""
CNN μοντέλο για ταξινόμηση εικόνων γάτας και σκύλου.

Αυτό το module περιέχει:
- Custom CNN αρχιτεκτονική
- Transfer learning με pre-trained models
- Grad-CAM υποστήριξη για ερμηνεύσιμη μάθηση
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


class CatDogClassifier(nn.Module):
    """
    CNN μοντέλο για ταξινόμηση εικόνων γάτας και σκύλου.
    
    Αυτή η κλάση υλοποιεί μια αρχιτεκτονική βασισμένη σε ResNet18
    με προσαρμογές για binary classification και υποστήριξη Grad-CAM.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, 
                 dropout_rate: float = 0.5):
        """
        Αρχικοποίηση του μοντέλου.
        
        Args:
            num_classes: Αριθμός κλάσεων (2 για cat/dog)
            pretrained: Χρήση pre-trained weights
            dropout_rate: Ποσοστό dropout για regularization
        """
        super(CatDogClassifier, self).__init__()
        
        # Φόρτωση pre-trained ResNet18 
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet18(weights=weights)
        
        # Αφαίρεση του τελευταίου fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Προσθήκη custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Grad-CAM attributes
        self.gradients = None
        self.activations = None
        
        # Εγγραφή hooks για Grad-CAM
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Εγγραφή hooks για την εξαγωγή gradients και activations για Grad-CAM.
        """
        def save_gradient(grad):
            self.gradients = grad
        
        def save_activation(module, input, output):
            self.activations = output
        
        # Εγγραφή hooks στο τελευταίο convolutional layer
        target_layer = self.backbone.layer4[-1]
        target_layer.register_forward_hook(save_activation)
        
        # Εγγραφή backward hook με έλεγχο για gradient
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None and grad_output[0].requires_grad:
                grad_output[0].register_hook(save_gradient)
        
        target_layer.register_backward_hook(backward_hook)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass του μοντέλου.
        
        Args:
            x: Είσοδος εικόνες (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Προβλέψεις κλάσεων
        """
        # Εξαγωγή features
        features = self.features(x)
        
        # Flatten για fully connected layers
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_gradcam_weights(self, target_class: int) -> torch.Tensor:
        """
        Υπολογισμός βαρών για Grad-CAM.
        
        Args:
            target_class: Η κλάση για την οποία υπολογίζονται τα βάρη
            
        Returns:
            torch.Tensor: Βάρη για Grad-CAM visualization
        """
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients ή activations δεν είναι διαθέσιμα. Εκτελέστε forward pass πρώτα.")
        
        # Υπολογισμός βαρών με gradient pooling
        weights = torch.mean(self.gradients, dim=[2, 3])
        
        return weights
    
    def get_activations(self) -> torch.Tensor:
        """
        Επιστροφή των activations του τελευταίου convolutional layer.
        
        Returns:
            torch.Tensor: Activations για Grad-CAM
        """
        if self.activations is None:
            raise ValueError("Activations δεν είναι διαθέσιμα. Εκτελέστε forward pass πρώτα.")
        
        return self.activations
    
    def reset_gradcam(self):
        """
        Επαναφορά των Grad-CAM attributes.
        """
        self.gradients = None
        self.activations = None


class CustomCNN(nn.Module):
    """
    Custom CNN αρχιτεκτονική από την αρχή.
    
    Αυτή η κλάση υλοποιεί μια απλή αλλά αποτελεσματική CNN
    για binary classification χωρίς transfer learning.
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Αρχικοποίηση του custom CNN.
        
        Args:
            num_classes: Αριθμός κλάσεων
            dropout_rate: Ποσοστό dropout
        """
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass του custom CNN.
        
        Args:
            x: Είσοδος εικόνες
            
        Returns:
            torch.Tensor: Προβλέψεις κλάσεων
        """
        # Convolutional layers με ReLU activation και batch normalization
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def create_model(model_type: str = "resnet18", num_classes: int = 2, 
                pretrained: bool = True, dropout_rate: float = 0.5) -> nn.Module:
    """
    Factory function για δημιουργία μοντέλων.
    
    Args:
        model_type: Τύπος μοντέλου ("resnet18", "custom")
        num_classes: Αριθμός κλάσεων
        pretrained: Χρήση pre-trained weights
        dropout_rate: Ποσοστό dropout
        
    Returns:
        nn.Module: Το δημιουργημένο μοντέλο
    """
    if model_type == "resnet18":
        return CatDogClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    elif model_type == "custom":
        return CustomCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Άγνωστος τύπος μοντέλου: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    Υπολογισμός αριθμού παραμέτρων του μοντέλου.
    
    Args:
        model: Το μοντέλο
        
    Returns:
        int: Αριθμός παραμέτρων
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """
    Δημιουργία σύνοψης του μοντέλου.
    
    Args:
        model: Το μοντέλο
        
    Returns:
        str: Σύνοψη του μοντέλου
    """
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
    Σύνοψη Μοντέλου:
    - Συνολικές παράμετροι: {total_params:,}
    - Εκπαιδεύσιμες παράμετροι: {trainable_params:,}
    - Αρχιτεκτονική: {model.__class__.__name__}
    """
    
    return summary 