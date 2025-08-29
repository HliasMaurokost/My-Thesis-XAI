
"""
Κύρια είσοδος της εφαρμογής ταξινόμησης εικόνων γάτας και σκύλου με XAI.

Αυτό το αρχείο παρέχει τη βασική διεπαφή για την εκπαίδευση μοντέλων,
την αξιολόγηση και την εφαρμογή τεχνικών ερμηνεύσιμης βαθιάς μάθησης.
"""

import argparse
import logging
import os
import sys
import torch
from pathlib import Path

# Προσθήκη του root directory στο path για imports
sys.path.append(str(Path(__file__).parent))

from utils.config import setup_logging, check_gpu_availability
from utils.data_utils import prepare_dataset_structure, validate_dataset, cleanup_corrupted_links
from training.trainer import ModelTrainer
from explainability.explainer_factory import ExplainerFactory


def parse_arguments():
    """
    Ανάλυση των command line arguments.
    
    Returns:
        argparse.Namespace: Τα parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Εφαρμογή ταξινόμησης εικόνων γάτας/σκύλου με XAI"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "explain"],
        default="train",
        help="Λειτουργία εκτέλεσης: train, evaluate, ή explain"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Dataset",
        help="Διαδρομή προς το dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Φάκελος εξόδου για αποτελέσματα"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Διαδρομή προς το αποθηκευμένο μοντέλο"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Αριθμός epochs για εκπαίδευση"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Μέγεθος batch"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--explain_method",
        type=str,
        choices=["gradcam", "lime", "shap", "all"],
        default="all",
        help="Μέθοδος εξήγησης XAI"
    )
    
    parser.add_argument(
        "--quick_train",
        action="store_true",
        help="Γρήγορη εκπαίδευση (3 epochs, μικρό batch size)"
    )
    
    parser.add_argument(
        "--full_train",
        action="store_true",
        help="Πλήρης εκπαίδευση (20 epochs, κανονικό batch size)"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Καθαρισμός κατεστραμμένων links από το dataset"
    )
    
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Μόνο validation του dataset χωρίς εκπαίδευση"
    )
    
    return parser.parse_args()


def get_training_choice():
    """
    Ερώτηση χρήστη για τον τύπο εκπαίδευσης.
    
    Returns:
        tuple: (epochs, batch_size, description)
    """
    print("=" * 80)
    print(" ΕΦΑΡΜΟΓΗ ΤΑΞΙΝΟΜΗΣΗΣ ΕΙΚΟΝΩΝ ΓΑΤΑΣ/ΣΚΥΛΟΥ")
    print("=" * 80)
    print()
    print("Επιλέξτε τον τύπο εκπαίδευσης:")
    print()
    print("1.  ΓΡΗΓΟΡΗ ΕΚΠΑΙΔΕΥΣΗ")
    print("   - 3 epochs")
    print("   - Batch size: 16")
    print("   - Χρόνος: ~5-10 λεπτά")
    print("   - Σκοπός: Δοκιμή λειτουργικότητας")
    print()
    print("2.  ΠΛΗΡΗΣ ΕΚΠΑΙΔΕΥΣΗ")
    print("   - 20 epochs")
    print("   - Batch size: 32")
    print("   - Χρόνος: ~30-60 λεπτά")
    print("   - Σκοπός: Πλήρης εκπαίδευση μοντέλου")
    print()
    print("3.   ΠΡΟΣΑΡΜΟΣΜΕΝΗ ΕΚΠΑΙΔΕΥΣΗ")
    print("   - Επιλογή παραμέτρων από command line")
    print()
    
    while True:
        choice = input("Επιλογή (1-3): ").strip()
        
        if choice == "1":
            return 3, 16, "Γρήγορη εκπαίδευση"
        elif choice == "2":
            return 20, 32, "Πλήρης εκπαίδευση"
        elif choice == "3":
            print("Χρησιμοποίησε τα command line arguments για προσαρμοσμένη εκπαίδευση.")
            print("Παράδειγμα: python main.py --mode train --epochs 15 --batch_size 64")
            return None, None, None
        else:
            print(" Λάθος επιλογή. Επιλέξτε 1, 2, ή 3.")


def main():
    """
    Κύρια συνάρτηση της εφαρμογής.
    
    Διαχειρίζεται τη ροή εκτέλεσης βάσει των command line arguments
    και καλεί τις κατάλληλες λειτουργίες με robust error handling.
    """
    try:
        # Ανάλυση arguments
        args = parse_arguments()
        
        # Έλεγχος για γρήγορη ή πλήρη εκπαίδευση
        if args.mode == "train" and not args.quick_train and not args.full_train:
            # Ερώτηση χρήστη για τον τύπο εκπαίδευσης
            training_choice = get_training_choice()
            
            if training_choice[0] is None:
                # Χρήστης επέλεξε προσαρμοσμένη εκπαίδευση
                print("Εκτέλεση με command line arguments...")
            else:
                # Εφαρμογή επιλογής χρήστη
                epochs, batch_size, description = training_choice
                args.epochs = epochs
                args.batch_size = batch_size
                
                print(f"\n Επιλέχθηκε: {description}")
                print(f"   Epochs: {args.epochs}")
                print(f"   Batch size: {args.batch_size}")
                print()
        
        # Ρύθμιση logging
        setup_logging(args.output_dir)
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 80)
        logger.info("ΕΦΑΡΜΟΓΗ ΤΑΞΙΝΟΜΗΣΗΣ ΕΙΚΟΝΩΝ ΓΑΤΑΣ/ΣΚΥΛΟΥ")
        logger.info("=" * 80)
        logger.info(f"Λειτουργία: {args.mode}")
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Έλεγχος διαθεσιμότητας GPU
        device = check_gpu_availability()
        logger.info(f"Χρήση συσκευής: {device}")
        
        # Ενημερωτικό μήνυμα για τη συσκευή που θα χρησιμοποιηθεί
        if device.type == "cuda":
            print("\n" + "=" * 60)
            print("  ΕΚΤΕΛΕΣΗ ΜΕ CUDA GPU")
            print("=" * 60)
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            print(f" CUDA Version: {torch.version.cuda}")
            print(f" PyTorch Version: {torch.__version__}")
            print(" Εκπαίδευση θα είναι γρήγορη με GPU acceleration")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("   ΕΚΤΕΛΕΣΗ ΜΕ CPU")
            print("=" * 60)
            print(" GPU δεν βρέθηκε ή δεν είναι διαθέσιμο")
            print(" Εκπαίδευση θα είναι αργή με CPU")
            print(" Προτείνεται GPU για καλύτερη απόδοση")
            print("=" * 60)
        
        # Δημιουργία φακέλων εξόδου με robust error handling
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs("models", exist_ok=True)
            logger.info("Φάκελοι εξόδου δημιουργήθηκαν επιτυχώς")
        except Exception as e:
            logger.error(f"Σφάλμα στη δημιουργία φακέλων: {e}")
            raise
        
        # Έλεγχος ύπαρξης dataset
        dataset_path_obj = Path(args.dataset_path)
        if not dataset_path_obj.exists():
            raise FileNotFoundError(f"Το dataset path δεν υπάρχει: {args.dataset_path}")
        
        # Προετοιμασία δομής dataset με robust error handling
        logger.info("Προετοιμασία dataset...")
        try:
            prepare_dataset_structure(args.dataset_path)
            validate_dataset(args.dataset_path)
            logger.info("Dataset προετοιμάστηκε επιτυχώς")
        except Exception as e:
            logger.error(f"Σφάλμα στην προετοιμασία dataset: {e}")
            logger.error("Προσπαθήστε να ελέγξετε τη διαδρομή του dataset")
            raise
        
        # Καθαρισμός κατεστραμμένων links αν ζητηθεί
        if args.cleanup:
            logger.info("Καθαρισμός κατεστραμμένων links...")
            try:
                cleanup_corrupted_links("dataset")
                logger.info("Καθαρισμός ολοκληρώθηκε επιτυχώς")
            except Exception as e:
                logger.error(f"Σφάλμα στον καθαρισμό: {e}")
                raise
        
        # Validation μόνο αν ζητηθεί
        if args.validate_only:
            logger.info("Validation dataset...")
            try:
                if validate_dataset(args.dataset_path):
                    logger.info("Dataset validation επιτυχής")
                    print("\n" + "=" * 60)
                    print(" DATASET VALIDATION ΕΠΙΤΥΧΗΣ")
                    print("=" * 60)
                    print(" Το dataset είναι έγκυρο και έτοιμο για εκπαίδευση")
                    print("=" * 60)
                else:
                    logger.error("Dataset validation απέτυχε")
                    raise RuntimeError("Dataset validation απέτυχε")
            except Exception as e:
                logger.error(f"Σφάλμα στο validation: {e}")
                raise
            return
        
        # Αρχικοποίηση trainer με robust error handling
        try:
            trainer = ModelTrainer(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device
            )
            logger.info("Trainer αρχικοποιήθηκε επιτυχώς")
        except Exception as e:
            logger.error(f"Σφάλμα στην αρχικοποίηση trainer: {e}")
            raise
        
        # Εκτέλεση λειτουργίας βάσει mode
        if args.mode == "train":
            logger.info("=" * 60)
            logger.info("ΈΝΑΡΞΗ ΕΚΠΑΙΔΕΥΣΗΣ ΜΟΝΤΕΛΟΥ")
            logger.info("=" * 60)
            logger.info(f"Epochs: {args.epochs}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Learning rate: {args.learning_rate}")
            
            try:
                trainer.train(epochs=args.epochs)
                logger.info("Εκπαίδευση ολοκληρώθηκε επιτυχώς")
                
                # Προβολή αποτελεσμάτων
                print("\n" + "=" * 80)
                print(" ΕΚΠΑΙΔΕΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ!")
                print("=" * 80)
                print(f" Αποτελέσματα στο φάκελο: {args.output_dir}")
                print(f" Μετρικές: {args.output_dir}/metrics/")
                print(f" Γραφήματα: {args.output_dir}/plots/")
                print(f" Μοντέλα: {args.output_dir}/models/")
                print()
                print(" Για να δεις τα αποτελέσματα εκτέλεσε:")
                print("   python show_results.py")
                print("=" * 80)
                
            except Exception as e:
                logger.error(f"Σφάλμα κατά την εκπαίδευση: {e}")
                raise
            
        elif args.mode == "evaluate":
            logger.info("Έναρξη αξιολόγησης μοντέλου")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Το μοντέλο δεν βρέθηκε: {args.model_path}")
            
            try:
                trainer.evaluate(args.model_path)
                logger.info("Αξιολόγηση ολοκληρώθηκε επιτυχώς")
            except Exception as e:
                logger.error(f"Σφάλμα κατά την αξιολόγηση: {e}")
                raise
                
        elif args.mode == "explain":
            logger.info("Έναρξη ανάλυσης ερμηνεύσιμης μάθησης")
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Το μοντέλο δεν βρέθηκε: {args.model_path}")
            
            try:
                explainer_factory = ExplainerFactory(
                    model_path=args.model_path,
                    dataset_path=args.dataset_path,
                    output_dir=args.output_dir,
                    device=device
                )
                
                if args.explain_method == "all":
                    methods = ["gradcam", "lime", "shap"]
                else:
                    methods = [args.explain_method]
                    
                for method in methods:
                    logger.info(f"Εφαρμογή {method.upper()}")
                    try:
                        explainer = explainer_factory.create_explainer(method)
                        explainer.explain_sample()
                        logger.info(f"{method.upper()} ολοκληρώθηκε επιτυχώς")
                    except Exception as e:
                        logger.error(f"Σφάλμα στο {method.upper()}: {e}")
                        continue
                    
                logger.info("Ανάλυση XAI ολοκληρώθηκε")
            except Exception as e:
                logger.error(f"Σφάλμα κατά την ανάλυση XAI: {e}")
                raise
        
        logger.info("=" * 80)
        logger.info("ΕΦΑΡΜΟΓΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n Εκτέλεση διακόπηκε από τον χρήστη")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n ΣΦΑΛΜΑ: {e}")
        print(" Ελέγξτε τα logs για περισσότερες λεπτομέρειες")
        sys.exit(1)


if __name__ == "__main__":
    main() 