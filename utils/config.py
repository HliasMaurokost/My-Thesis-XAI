
"""
Ρύθμιση logging και configuration για την εφαρμογή.

Αυτό το module περιέχει συναρτήσεις για:
- Ρύθμιση logging με διαφορετικά επίπεδα
- Έλεγχο διαθεσιμότητας GPU
- Βασικές ρυθμίσεις εφαρμογής
"""

import logging
import os
import torch
from datetime import datetime
from pathlib import Path


def setup_logging(output_dir: str, level: str = "INFO") -> None:
    """
    Ρύθμιση του logging συστήματος.
    
    Args:
        output_dir: Φάκελος εξόδου για τα log files
        level: Επίπεδο logging (DEBUG, INFO, WARNING, ERROR)
    """
    # Δημιουργία φακέλου εξόδου αν δεν υπάρχει
    os.makedirs(output_dir, exist_ok=True)
    
    # Ρύθμιση format για τα logs
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Ρύθμιση βασικού logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(
                os.path.join(output_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                encoding='utf-8'
            )
        ]
    )
    
    # Καταστολή των logs από external libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def check_gpu_availability() -> torch.device:
    """
    Έλεγχος διαθεσιμότητας GPU και επιστροφή κατάλληλης συσκευής.
    
    Returns:
        torch.device: Η συσκευή που θα χρησιμοποιηθεί (cuda ή cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU διαθέσιμο: {torch.cuda.get_device_name(0)}")
        logging.info(f"Αριθμός CUDA cores: {torch.cuda.device_count()}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logging.warning("GPU δεν είναι διαθέσιμο. Χρήση CPU.")
    
    return device


def get_model_config() -> dict:
    """
    Επιστροφή βασικών ρυθμίσεων για το μοντέλο.
    
    Returns:
        dict: Λεξικό με τις ρυθμίσεις του μοντέλου
    """
    return {
        "input_size": (224, 224),
        "num_classes": 2,
        "class_names": ["cat", "dog"],
        "image_mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "image_std": [0.229, 0.224, 0.225],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 10,
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "early_stopping_patience": 5
    }


def get_training_config() -> dict:
    """
    Επιστροφή ρυθμίσεων για την εκπαίδευση.
    
    Returns:
        dict: Λεξικό με τις ρυθμίσεις εκπαίδευσης
    """
    return {
        "optimizer": "adam",
        "scheduler": "step",
        "scheduler_step_size": 7,
        "scheduler_gamma": 0.1,
        "criterion": "cross_entropy",
        "save_best_model": True,
        "save_checkpoint_interval": 5,
        "tensorboard_logging": True,
        "gradient_clipping": 1.0
    }


def get_augmentation_config() -> dict:
    """
    Επιστροφή ρυθμίσεων για data augmentation.
    
    Returns:
        dict: Λεξικό με τις ρυθμίσεις augmentation
    """
    return {
        "horizontal_flip_prob": 0.5,
        "rotation_degrees": 15,
        "brightness_factor": 0.2,
        "contrast_factor": 0.2,
        "saturation_factor": 0.2,
        "hue_factor": 0.1,
        "resize_size": (256, 256),
        "crop_size": (224, 224),
        "normalize": True
    } 