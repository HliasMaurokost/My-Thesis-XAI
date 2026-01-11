
"""
Factory pattern για δημιουργία XAI explainers.

Αυτό το module παρέχει ένα factory για τη δημιουργία διαφορετικών
τύπων explainers (Grad-CAM, LIME, SHAP) με ενιαία διεπαφή.
"""

import logging
from typing import Dict, Type
from abc import ABC, abstractmethod

from .gradcam_explainer import GradCAMExplainer
from .lime_explainer import LIMEExplainer
from .shap_explainer import SHAPExplainer
from .rule_based_explainer import RuleBasedExplainer


class BaseExplainer(ABC):
    """
    Abstract base class για όλους τους explainers.
    
    Αυτή η κλάση ορίζει τη βασική διεπαφή που πρέπει να υλοποιήσουν
    όλοι οι explainers.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του base explainer.
        
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
    
    @abstractmethod
    def explain_sample(self, sample_idx: int = 0):
        """
        Εφαρμογή εξήγησης σε ένα δείγμα.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        pass
    
    @abstractmethod
    def explain_batch(self, batch_size: int = 5):
        """
        Εφαρμογή εξήγησης σε ένα batch δειγμάτων.
        
        Args:
            batch_size: Μέγεθος batch
        """
        pass


class ExplainerFactory:
    """
    Factory κλάση για δημιουργία XAI explainers.
    
    Αυτή η κλάση παρέχει μια ενιαία διεπαφή για τη δημιουργία
    διαφορετικών τύπων explainers.
    """
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str, device):
        """
        Αρχικοποίηση του factory.
        
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
        
        self.logger = logging.getLogger(__name__)
        
        # Registry των διαθέσιμων explainers
        self._explainers: Dict[str, Type[BaseExplainer]] = {
            "gradcam": GradCAMExplainer,
            "lime": LIMEExplainer,
            "shap": SHAPExplainer,
            "rule_based": RuleBasedExplainer
        }
    
    def create_explainer(self, explainer_type: str) -> BaseExplainer:
        """
        Δημιουργία explainer βάσει τύπου.
        
        Args:
            explainer_type: Τύπος explainer ("gradcam", "lime", "shap")
            
        Returns:
            BaseExplainer: Ο δημιουργημένος explainer
            
        Raises:
            ValueError: Αν ο τύπος explainer δεν είναι διαθέσιμος
        """
        if explainer_type not in self._explainers:
            available_types = list(self._explainers.keys())
            raise ValueError(
                f"Άγνωστος τύπος explainer: {explainer_type}. "
                f"Διαθέσιμοι τύποι: {available_types}"
            )
        
        explainer_class = self._explainers[explainer_type]
        
        self.logger.info(f"Δημιουργία {explainer_type.upper()} explainer")
        
        return explainer_class(
            model_path=self.model_path,
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            device=self.device
        )
    
    def get_available_explainers(self) -> list:
        """
        Επιστροφή λίστας διαθέσιμων explainers.
        
        Returns:
            list: Λίστα με τους διαθέσιμους τύπους explainers
        """
        return list(self._explainers.keys())
    
    def register_explainer(self, name: str, explainer_class: Type[BaseExplainer]):
        """
        Εγγραφή νέου τύπου explainer.
        
        Args:
            name: Όνομα του explainer
            explainer_class: Κλάση του explainer
        """
        if not issubclass(explainer_class, BaseExplainer):
            raise ValueError("Ο explainer πρέπει να κληρονομεί από BaseExplainer")
        
        self._explainers[name] = explainer_class
        self.logger.info(f"Εγγράφηκε νέος explainer: {name}")
    
    def explain_with_all_methods(self, sample_idx: int = 0):
        """
        Εφαρμογή όλων των διαθέσιμων μεθόδων εξήγησης.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        self.logger.info("Εφαρμογή όλων των μεθόδων XAI")
        
        for explainer_type in self._explainers.keys():
            try:
                explainer = self.create_explainer(explainer_type)
                explainer.explain_sample(sample_idx)
                self.logger.info(f"{explainer_type.upper()} ολοκληρώθηκε επιτυχώς")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο {explainer_type}: {e}")
    
    def compare_methods(self, sample_idx: int = 0):
        """
        Σύγκριση διαφορετικών μεθόδων εξήγησης.
        
        Args:
            sample_idx: Δείκτης του δείγματος
        """
        self.logger.info("Σύγκριση μεθόδων XAI")
        
        results = {}
        
        for explainer_type in self._explainers.keys():
            try:
                explainer = self.create_explainer(explainer_type)
                result = explainer.explain_sample(sample_idx)
                results[explainer_type] = result
                self.logger.info(f"{explainer_type.upper()}: {result}")
            except Exception as e:
                self.logger.error(f"Σφάλμα στο {explainer_type}: {e}")
                results[explainer_type] = None
        
        return results 