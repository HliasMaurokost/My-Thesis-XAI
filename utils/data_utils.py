
"""
Βοηθητικές συναρτήσεις για χειρισμό dataset και validation.

Αυτό το module περιέχει συναρτήσεις για:
- Προετοιμασία δομής dataset
- Έλεγχο κατεστραμμένων εικόνων
- Validation του dataset
- Μετατροπή ονομάτων φακέλων
"""

import os
import shutil
import logging
from pathlib import Path
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms.functional as TF


class RobustImageFolder(Dataset):
    """
    Custom dataset class που κάνει skip κατεστραμμένες εικόνες κατά την εκπαίδευση.
    
    Αυτή η κλάση επεκτείνει το ImageFolder με robust error handling
    για κατεστραμμένες εικόνες, αποφεύγοντας τη διακοπή της εκπαίδευσης.
    """
    
    def __init__(self, root, transform=None, target_transform=None):
        """
        Αρχικοποίηση του RobustImageFolder.
        
        Args:
            root: Διαδρομή προς το dataset
            transform: Transforms για τις εικόνες
            target_transform: Transforms για τα labels
        """
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        # Δημιουργία λίστας έγκυρων εικόνων
        self.valid_samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._build_valid_samples()
    
    def _build_valid_samples(self):
        """Δημιουργία λίστας μόνο των έγκυρων εικόνων."""
        logger = logging.getLogger(__name__)
        
        # Εύρεση φακέλων κλάσεων
        class_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        class_dirs.sort()  # Για σταθερή σειρά
        
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        logger.info(f"Βρέθηκαν κλάσεις: {self.classes}")
        
        # Έλεγχος κάθε εικόνας
        total_images = 0
        valid_images = 0
        
        for class_idx, class_dir in enumerate(class_dirs):
            logger.info(f"Έλεγχος κλάσης {class_dir.name}...")
            
            # Εύρεση όλων των αρχείων εικόνας
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = [f for f in class_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            total_images += len(image_files)
            
            for img_path in image_files:
                if self._is_valid_image(img_path):
                    self.valid_samples.append((img_path, class_idx))
                    valid_images += 1
                else:
                    logger.warning(f"Κατεστραμμένη εικόνα παραλείφθηκε: {img_path}")
        
        logger.info(f"Έγκυρες εικόνες: {valid_images}/{total_images}")
        
        if len(self.valid_samples) == 0:
            raise ValueError("Δεν βρέθηκαν έγκυρες εικόνες στο dataset")
    
    def _is_valid_image(self, img_path):
        """
        Έλεγχος αν μια εικόνα είναι έγκυρη και μπορεί να φορτωθεί.
        
        Args:
            img_path: Διαδρομή προς την εικόνα
            
        Returns:
            bool: True αν η εικόνα είναι έγκυρη
        """
        try:
            with Image.open(img_path) as img:
                # Έλεγχος αν η εικόνα μπορεί να φορτωθεί
                img.verify()
                
                # Έλεγχος αν η εικόνα μπορεί να μετατραπεί σε RGB
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Έλεγχος διαστάσεων
                if img.size[0] < 10 or img.size[1] < 10:
                    return False
                
                return True
                
        except Exception:
            return False
    
    def __len__(self):
        """Επιστροφή αριθμού έγκυρων εικόνων."""
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """
        Φόρτωση εικόνας με robust error handling.
        
        Args:
            idx: Δείκτης εικόνας
            
        Returns:
            tuple: (εικόνα, label)
        """
        img_path, class_idx = self.valid_samples[idx]
        
        try:
            # Φόρτωση εικόνας
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Εφαρμογή transforms
                if self.transform is not None:
                    img = self.transform(img)
                
                # Εφαρμογή target transforms
                if self.target_transform is not None:
                    class_idx = self.target_transform(class_idx)
                
                return img, class_idx
                
        except Exception as e:
            # Σε περίπτωση σφάλματος, επιστρέφουμε την πρώτη έγκυρη εικόνα
            logging.getLogger(__name__).warning(f"Σφάλμα στη φόρτωση {img_path}: {e}")
            return self.__getitem__(0)  # Επιστροφή πρώτης εικόνας


def prepare_dataset_structure(dataset_path: str) -> None:
    """
    Προετοιμασία της δομής του dataset για χρήση με ImageFolder.
    
    Δημιουργεί symbolic links προς το αρχικό dataset χωρίς να το αλλάξει.
    Robust validation και error handling για επαγγελματική ποιότητα.
    
    Args:
        dataset_path: Διαδρομή προς το αρχικό dataset
    """
    logger = logging.getLogger(__name__)
    
    # Δημιουργία φακέλου dataset αν δεν υπάρχει
    dataset_dir = Path("dataset")
    
    # Δημιουργία φακέλων για κάθε κλάση
    cat_dir = dataset_dir / "Cat"
    dog_dir = dataset_dir / "Dog"
    
    # Έλεγχος αν η προετοιμασία έχει ήδη γίνει και είναι έγκυρη
    if _is_dataset_valid(cat_dir, dog_dir):
        logger.info("Dataset ήδη προετοιμασμένο και έγκυρο")
        return
    
    # Καθαρισμός και δημιουργία φακέλων
    import shutil
    if dataset_dir.exists():
        logger.info("Καθαρισμός υπάρχοντος dataset...")
        shutil.rmtree(dataset_dir)
    
    dataset_dir.mkdir(exist_ok=True)
    cat_dir.mkdir(exist_ok=True)
    dog_dir.mkdir(exist_ok=True)
    
    # Διαδρομές προς τους αρχικούς φακέλους
    original_cat_path = Path(dataset_path) / "Cat"
    original_dog_path = Path(dataset_path) / "Dog"
    
    # Έλεγχος ύπαρξης αρχικών φακέλων
    if not original_cat_path.exists():
        raise FileNotFoundError(f"Ο φάκελος {original_cat_path} δεν βρέθηκε")
    if not original_dog_path.exists():
        raise FileNotFoundError(f"Ο φάκελος {original_dog_path} δεν βρέθηκε")
    
    logger.info("=" * 60)
    logger.info("ΠΡΟΕΤΟΙΜΑΣΙΑ DATASET")
    logger.info("=" * 60)
    
    logger.info("Δημιουργία symbolic links για εικόνες γάτας...")
    _create_symbolic_links(original_cat_path, cat_dir, "cat")
    
    logger.info("Δημιουργία symbolic links για εικόνες σκύλου...")
    _create_symbolic_links(original_dog_path, dog_dir, "dog")
    
    # Τελικός έλεγχος ποιότητας
    if _is_dataset_valid(cat_dir, dog_dir):
        logger.info("=" * 60)
        logger.info("ΠΡΟΕΤΟΙΜΑΣΙΑ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
        logger.info("=" * 60)
    else:
        raise RuntimeError("Η προετοιμασία του dataset απέτυχε")


def _is_dataset_valid(cat_dir: Path, dog_dir: Path) -> bool:
    """
    Έλεγχος εγκυρότητας του προετοιμασμένου dataset.
    
    Args:
        cat_dir: Φάκελος με εικόνες γάτας
        dog_dir: Φάκελος με εικόνες σκύλου
        
    Returns:
        bool: True αν το dataset είναι έγκυρο
    """
    logger = logging.getLogger(__name__)
    
    # Έλεγχος ύπαρξης φακέλων
    if not cat_dir.exists() or not dog_dir.exists():
        logger.warning("Φάκελοι dataset δεν υπάρχουν")
        return False
    
    # Έλεγχος αριθμού εικόνων
    cat_images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.jpeg")) + \
                 list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.bmp"))
    dog_images = list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.jpeg")) + \
                 list(dog_dir.glob("*.png")) + list(dog_dir.glob("*.bmp"))
    
    if len(cat_images) == 0 or len(dog_images) == 0:
        logger.warning(f"Ένας ή περισσότεροι φάκελοι είναι άδειοι (cat: {len(cat_images)}, dog: {len(dog_images)})")
        return False
    
    # Έλεγχος ελάχιστου αριθμού εικόνων
    min_images = 1000  # Ελάχιστος αριθμός εικόνων ανά κλάση
    if len(cat_images) < min_images or len(dog_images) < min_images:
        logger.warning(f"Πολύ λίγες εικόνες (cat: {len(cat_images)}, dog: {len(dog_images)}, ελάχιστο: {min_images})")
        return False
    
    logger.info(f"Dataset validation επιτυχής: cat={len(cat_images)}, dog={len(dog_images)}")
    return True


def cleanup_corrupted_links(dataset_dir: str = "dataset") -> None:
    """
    Καθαρισμός κατεστραμμένων symbolic links από το dataset.
    
    Αυτή η συνάρτηση ελέγχει και διαγράφει links που δεν δείχνουν σε έγκυρες εικόνες.
    
    Args:
        dataset_dir: Διαδρομή προς το dataset
    """
    logger = logging.getLogger(__name__)
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logger.warning(f"Ο φάκελος {dataset_dir} δεν υπάρχει")
        return
    
    cleaned_count = 0
    total_links = 0
    
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            logger.info(f"Έλεγχος φακέλου: {class_dir.name}")
            
            for link_file in class_dir.iterdir():
                if link_file.is_file():
                    total_links += 1
                    
                    try:
                        # Έλεγχος αν το link δείχνει σε έγκυρη εικόνα
                        if link_file.is_symlink():
                            target_path = link_file.resolve()
                            
                            # Έλεγχος αν το target υπάρχει και είναι έγκυρη εικόνα
                            if not target_path.exists():
                                logger.warning(f"Broken link: {link_file} -> {target_path}")
                                link_file.unlink()
                                cleaned_count += 1
                                continue
                            
                            # Έλεγχος αν η εικόνα μπορεί να φορτωθεί
                            with Image.open(target_path) as img:
                                img.verify()
                                img.load()
                                
                        else:
                            # Έλεγχος αν το αρχείο είναι έγκυρη εικόνα
                            with Image.open(link_file) as img:
                                img.verify()
                                img.load()
                                
                    except Exception as e:
                        logger.warning(f"Κατεστραμμένη εικόνα: {link_file} - {e}")
                        link_file.unlink()
                        cleaned_count += 1
    
    logger.info(f"Καθαρισμός ολοκληρώθηκε: διαγράφηκαν {cleaned_count}/{total_links} κατεστραμμένα links")


def _create_symbolic_links(source_dir: Path, target_dir: Path, class_name: str) -> None:
    """
    Δημιουργία symbolic links προς εικόνες από τον πηγαίο στον στόχος φάκελο.
    
    Robust έλεγχος εικόνων με πλήρη validation για αποφυγή σφαλμάτων κατά την εκπαίδευση.
    
    Args:
        source_dir: Πηγαίος φάκελος
        target_dir: Στόχος φάκελος
        class_name: Όνομα κλάσης για logging
    """
    logger = logging.getLogger(__name__)
    
    # Λίστα όλων των αρχείων εικόνας
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in source_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    logger.info(f"Βρέθηκαν {len(image_files)} εικόνες {class_name}")
    
    linked_count = 0
    corrupted_count = 0
    skipped_count = 0
    
    for i, image_file in enumerate(image_files):
        try:
            # Πλήρης έλεγχος εικόνας - verify + load για να διασφαλιστεί ότι είναι φορτώσιμη
            with Image.open(image_file) as img:
                # Έλεγχος αν η εικόνα μπορεί να επαληθευθεί
                img.verify()
                
                # Έλεγχος αν η εικόνα μπορεί να φορτωθεί πλήρως
                img.load()
                
                # Έλεγχος διαστάσεων (αποφυγή πολύ μικρών ή πολύ μεγάλων εικόνων)
                width, height = img.size
                if width < 10 or height < 10 or width > 10000 or height > 10000:
                    logger.warning(f"Εικόνα με μη αποδεκτές διαστάσεις: {image_file} ({width}x{height})")
                    skipped_count += 1
                    continue
                
                # Έλεγχος mode (αποφυγή εικόνων που δεν είναι RGB)
                if img.mode not in ['RGB', 'L']:
                    logger.warning(f"Εικόνα με μη αποδεκτό mode: {image_file} (mode: {img.mode})")
                    skipped_count += 1
                    continue
            
            # Δημιουργία symbolic link μόνο για έγκυρες εικόνες
            target_file = target_dir / image_file.name
            if not target_file.exists():
                target_file.symlink_to(image_file)
            linked_count += 1
            
            # Progress logging κάθε 1000 εικόνες
            if (i + 1) % 1000 == 0:
                logger.info(f"Επεξεργάστηκαν {i + 1}/{len(image_files)} εικόνες {class_name} "
                          f"(έγκυρες: {linked_count}, κατεστραμμένες: {corrupted_count}, παραλειφθείσες: {skipped_count})")
                
        except (OSError, IOError, SyntaxError, ValueError) as e:
            # Σφάλματα που σχετίζονται με κατεστραμμένες εικόνες
            logger.warning(f"Κατεστραμμένη εικόνα: {image_file} - {e}")
            corrupted_count += 1
            continue
        except Exception as e:
            # Άλλα σφάλματα
            logger.warning(f"Απρόσμενο σφάλμα για {image_file}: {e}")
            skipped_count += 1
            continue
    
    logger.info(f"Ολοκληρώθηκε επεξεργασία εικόνων {class_name}:")
    logger.info(f"  - Έγκυρες εικόνες: {linked_count}")
    logger.info(f"  - Κατεστραμμένες εικόνες: {corrupted_count}")
    logger.info(f"  - Παραλειφθείσες εικόνες: {skipped_count}")
    logger.info(f"  - Συνολική επιτυχία: {linked_count}/{len(image_files)} ({linked_count/len(image_files)*100:.1f}%)")


def _copy_images(source_dir: Path, target_dir: Path, class_name: str) -> None:
    """
    Αντιγραφή εικόνων από τον πηγαίο στον στόχος φάκελο.
    
    Args:
        source_dir: Πηγαίος φάκελος
        target_dir: Στόχος φάκελος
        class_name: Όνομα κλάσης για logging
    """
    logger = logging.getLogger(__name__)
    
    # Λίστα όλων των αρχείων εικόνας
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in source_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    logger.info(f"Βρέθηκαν {len(image_files)} εικόνες {class_name}")
    
    copied_count = 0
    for i, image_file in enumerate(image_files):
        try:
            # Έλεγχος αν η εικόνα είναι έγκυρη
            with Image.open(image_file) as img:
                img.verify()
            
            # Αντιγραφή αρχείου
            target_file = target_dir / image_file.name
            shutil.copy2(image_file, target_file)
            copied_count += 1
            
            # Progress logging κάθε 1000 εικόνες
            if (i + 1) % 1000 == 0:
                logger.info(f"Αντιγράφηκαν {i + 1}/{len(image_files)} εικόνες {class_name}")
                
        except Exception as e:
            logger.warning(f"Σφάλμα στην αντιγραφή {image_file}: {e}")
            continue
    
    logger.info(f"Ολοκληρώθηκε αντιγραφή {copied_count}/{len(image_files)} εικόνων {class_name}")


def check_corrupt_images(dataset_path: str) -> list:
    """
    Έλεγχος για κατεστραμμένες εικόνες στο dataset.
    
    Args:
        dataset_path: Διαδρομή προς το dataset
        
    Returns:
        list: Λίστα με τα paths των κατεστραμμένων εικόνων
    """
    logger = logging.getLogger(__name__)
    corrupt_files = []
    
    dataset_dir = Path(dataset_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            logger.info(f"Έλεγχος φακέλου: {class_dir.name}")
            
            for image_file in class_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    try:
                        with Image.open(image_file) as img:
                            img.verify()
                            # Έλεγχος αν η εικόνα μπορεί να φορτωθεί
                            img.load()
                    except Exception as e:
                        logger.warning(f"Κατεστραμμένη εικόνα: {image_file} - {e}")
                        corrupt_files.append(str(image_file))
    
    logger.info(f"Βρέθηκαν {len(corrupt_files)} κατεστραμμένες εικόνες")
    return corrupt_files


def validate_dataset(dataset_path: str) -> bool:
    """
    Validation του dataset για έλεγχο ορθότητας δομής.
    
    Args:
        dataset_path: Διαδρομή προς το dataset
        
    Returns:
        bool: True αν το dataset είναι έγκυρο, False διαφορετικά
    """
    logger = logging.getLogger(__name__)
    
    # Για validation χρησιμοποιούμε το προετοιμασμένο dataset
    dataset_dir = Path("dataset")
    
    # Έλεγχος ύπαρξης φακέλου dataset
    if not dataset_dir.exists():
        logger.error(f"Ο φάκελος {dataset_dir} δεν υπάρχει")
        return False
    
    # Έλεγχος ύπαρξης φακέλων κλάσεων
    expected_classes = ["Cat", "Dog"]  # Διόρθωση: χρήση κεφαλαίων όπως στο πραγματικό dataset
    found_classes = []
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            found_classes.append(class_dir.name)
    
    missing_classes = set(expected_classes) - set(found_classes)
    if missing_classes:
        logger.error(f"Λείπουν οι φάκελοι κλάσεων: {missing_classes}")
        return False
    
    # Έλεγχος αριθμού εικόνων ανά κλάση
    for class_name in expected_classes:
        class_dir = dataset_dir / class_name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png")) + list(class_dir.glob("*.bmp"))
        
        logger.info(f"Κλάση {class_name}: {len(image_files)} εικόνες")
        
        if len(image_files) == 0:
            logger.error(f"Δεν βρέθηκαν εικόνες στην κλάση {class_name}")
            return False
    
    logger.info("Dataset validation επιτυχής")
    return True


def create_data_loaders(dataset_path: str, batch_size: int = 32, 
                       train_split: float = 0.7, val_split: float = 0.15, 
                       test_split: float = 0.15, random_seed: int = 42) -> tuple:
    """
    Δημιουργία DataLoaders για εκπαίδευση, validation και test.
    
    Robust error handling και validation για επαγγελματική ποιότητα.
    
    Args:
        dataset_path: Διαδρομή προς το dataset
        batch_size: Μέγεθος batch
        train_split: Ποσοστό για training set (προεπιλογή: 0.7)
        val_split: Ποσοστό για validation set (προεπιλογή: 0.15)
        test_split: Ποσοστό για test set (προεπιλογή: 0.15)
        random_seed: Seed για reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    logger = logging.getLogger(__name__)
    
    # Validation των παραμέτρων
    if not (0 < train_split < 1 and 0 < val_split < 1 and 0 < test_split < 1):
        raise ValueError("Τα ποσοστά splits πρέπει να είναι μεταξύ 0 και 1")
    
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Τα ποσοστά splits πρέπει να αθροίζονται σε 1.0")
    
    if batch_size <= 0:
        raise ValueError("Το batch_size πρέπει να είναι θετικό")
    
    # Έλεγχος ύπαρξης dataset
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Το dataset path δεν υπάρχει: {dataset_path}")
    
    # Ρύθμιση transforms με robust error handling
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Φόρτωση dataset με robust error handling
    try:
        logger.info(f"Φόρτωση dataset από: {dataset_path}")
        full_dataset = RobustImageFolder(dataset_path, transform=train_transforms)
        
        if len(full_dataset) == 0:
            raise ValueError("Το dataset είναι άδειο")
        
        logger.info(f"Φορτώθηκε dataset με {len(full_dataset)} εικόνες")
        logger.info(f"Κλάσεις: {full_dataset.classes}")
        
        # Έλεγχος ότι υπάρχουν τουλάχιστον 2 κλάσεις
        if len(full_dataset.classes) < 2:
            raise ValueError(f"Το dataset πρέπει να έχει τουλάχιστον 2 κλάσεις, βρέθηκαν: {len(full_dataset.classes)}")
        
    except Exception as e:
        logger.error(f"Σφάλμα στη φόρτωση dataset: {e}")
        logger.error("Προσπαθήστε να τρέξετε ξανά την προετοιμασία του dataset")
        raise
    
    # Διαχωρισμός σε train/validation/test με validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Έλεγχος ότι τα μεγέθη είναι θετικά και λογικά
    if train_size <= 0:
        raise ValueError(f"Το training set είναι πολύ μικρό: {train_size} samples")
    if val_size <= 0:
        raise ValueError(f"Το validation set είναι πολύ μικρό: {val_size} samples")
    if test_size <= 0:
        raise ValueError(f"Το test set είναι πολύ μικρό: {test_size} samples")
    
    logger.info(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    # Δημιουργία splits με robust error handling
    try:
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
    except Exception as e:
        logger.error(f"Σφάλμα στη δημιουργία splits: {e}")
        raise
    
    # Εφαρμογή validation transforms στο validation και test set
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    # Δημιουργία DataLoaders με robust error handling
    try:
        # Προσαρμογή num_workers ανάλογα με το σύστημα
        import multiprocessing
        num_workers = min(4, multiprocessing.cpu_count())
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
    except Exception as e:
        logger.error(f"Σφάλμα στη δημιουργία DataLoaders: {e}")
        raise
    
    logger.info("=" * 50)
    logger.info("DATALOADERS ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ ΕΠΙΤΥΧΩΣ")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Workers: {num_workers}")
    logger.info("=" * 50)
    
    return train_loader, val_loader, test_loader, full_dataset.classes 