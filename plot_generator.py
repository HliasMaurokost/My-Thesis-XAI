
"""
Δυναμικός δημιουργός γραφημάτων για την εφαρμογή XAI.

Αυτό το module παρέχει λειτουργίες για τη δημιουργία γραφημάτων
με βάση τα πραγματικά δεδομένα της εφαρμογής.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Ρύθμιση για ελληνικά γραφήματα
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class DynamicPlotGenerator:
    """
    Κλάση για δυναμική δημιουργία γραφημάτων με βάση τα πραγματικά δεδομένα.
    """
    
    def __init__(self, output_dir: str = "outputs/plots"):
        """
        Αρχικοποίηση του δυναμικού δημιουργού γραφημάτων.
        
        Args:
            output_dir: Φάκελος εξόδου για τα γραφήματα
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Χρώματα για τα γραφήματα
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'light': '#F7F7F7',
            'dark': '#2C2C2C'
        }
        
        # Ρύθμιση στυλ
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_training_plots_from_data(self, training_data: dict):
        """
        Δημιουργία γραφημάτων εκπαίδευσης από πραγματικά δεδομένα.
        
        Args:
            training_data: Λεξικό με τα δεδομένα εκπαίδευσης
        """
        if not training_data:
            print("Δεν υπάρχουν δεδομένα εκπαίδευσης για τη δημιουργία γραφημάτων")
            return
        
        epochs = range(1, len(training_data.get('train_losses', [])) + 1)
        train_losses = training_data.get('train_losses', [])
        val_losses = training_data.get('val_losses', [])
        train_accuracies = training_data.get('train_accuracies', [])
        val_accuracies = training_data.get('val_accuracies', [])
        learning_rates = training_data.get('learning_rates', [])
        
        if not epochs:
            print("Δεν υπάρχουν δεδομένα για τη δημιουργία γραφημάτων εκπαίδευσης")
            return
        
        # Γράφημα απώλειας εκπαίδευσης και επικύρωσης
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        plt.title('Εκπαίδευση και Επικύρωση - Απώλεια', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Γράφημα ακρίβειας εκπαίδευσης και επικύρωσης
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2, marker='o')
        plt.plot(epochs, val_accuracies, 'm-', label='Validation Accuracy', linewidth=2, marker='s')
        plt.title('Εκπαίδευση και Επικύρωση - Ακρίβεια', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Γράφημα ρυθμού μάθησης
        if learning_rates:
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, learning_rates, 'orange', linewidth=2, marker='o')
            plt.title('Πρόγραμμα Ρυθμού Μάθησης', fontsize=16, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Ανάλυση διαφοράς απώλειας
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, loss_diff, 'purple', linewidth=2, marker='o')
        plt.title('Ανάλυση Διαφοράς Απώλειας (Train - Val)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Difference', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_difference_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Γραφήματα εκπαίδευσης δημιουργήθηκαν από πραγματικά δεδομένα")
    
    def create_performance_plots_from_data(self, performance_data: dict):
        """
        Δημιουργία γραφημάτων απόδοσης από πραγματικά δεδομένα.
        
        Args:
            performance_data: Λεξικό με τα δεδομένα απόδοσης
        """
        if not performance_data:
            print("Δεν υπάρχουν δεδομένα απόδοσης για τη δημιουργία γραφημάτων")
            return
        
        # Γράφημα μετρικών απόδοσης
        metrics = performance_data.get('metrics', {})
        if metrics:
            plt.figure(figsize=(12, 8))
            bars = plt.bar(metrics.keys(), metrics.values(), color=self.colors['primary'], alpha=0.7)
            plt.title('Μετρικές Απόδοσης Μοντέλου', fontsize=16, fontweight='bold')
            plt.xlabel('Μετρική', fontsize=12)
            plt.ylabel('Τιμή', fontsize=12)
            plt.ylim(0, 1)
            
            # Προσθήκη τιμών πάνω από τα bars
            for bar, value in zip(bars, metrics.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Πίνακας σύγχυσης
        confusion_matrix = performance_data.get('confusion_matrix')
        if confusion_matrix is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Πίνακας Σύγχυσης', fontsize=16, fontweight='bold')
            plt.xlabel('Προβλεπόμενη Κλάση', fontsize=12)
            plt.ylabel('Πραγματική Κλάση', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Καμπύλη ROC
        roc_data = performance_data.get('roc_curve')
        if roc_data and 'fpr' in roc_data and 'tpr' in roc_data:
            plt.figure(figsize=(10, 8))
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    color=self.colors['primary'], linewidth=2, 
                    label=f'ROC (AUC = {roc_data.get("auc", 0.0):.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            plt.title('ROC Curve', fontsize=16, fontweight='bold')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Γραφήματα απόδοσης δημιουργήθηκαν από πραγματικά δεδομένα")
    
    def create_dataset_plots_from_data(self, dataset_info: dict):
        """
        Δημιουργία γραφημάτων dataset από πραγματικά δεδομένα.
        
        Args:
            dataset_info: Λεξικό με πληροφορίες dataset
        """
        if not dataset_info:
            print("Δεν υπάρχουν πληροφορίες dataset για τη δημιουργία γραφημάτων")
            return
        
        # Κατανομή δειγμάτων
        split_sizes = dataset_info.get('split_sizes', {})
        if split_sizes:
            plt.figure(figsize=(10, 8))
            sizes = list(split_sizes.values())
            labels = list(split_sizes.keys())
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
            
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Κατανομή Δειγμάτων Dataset', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.savefig(os.path.join(self.output_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Κατανομή κλάσεων
        class_distribution = dataset_info.get('class_distribution', {})
        if class_distribution:
            plt.figure(figsize=(10, 8))
            classes = list(class_distribution.keys())
            counts = list(class_distribution.values())
            
            bars = plt.bar(classes, counts, color=[self.colors['primary'], self.colors['secondary']])
            plt.title('Κατανομή Κλάσεων', fontsize=16, fontweight='bold')
            plt.xlabel('Κλάση', fontsize=12)
            plt.ylabel('Αριθμός Δειγμάτων', fontsize=12)
            
            # Προσθήκη αριθμών πάνω από τα bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Γραφήματα dataset δημιουργήθηκαν από πραγματικά δεδομένα")
    
    def create_model_architecture_plot(self, model_info: dict):
        """
        Δημιουργία γραφήματος αρχιτεκτονικής μοντέλου.
        
        Args:
            model_info: Λεξικό με πληροφορίες μοντέλου
        """
        if not model_info:
            print("Δεν υπάρχουν πληροφορίες μοντέλου για τη δημιουργία γραφήματος")
            return
        
        layers = model_info.get('layers', [])
        parameters = model_info.get('parameters', [])
        activations = model_info.get('activations', [])
        
        if not layers or not parameters:
            print("Δεν υπάρχουν επαρκή δεδομένα για τη δημιουργία γραφήματος αρχιτεκτονικής")
            return
        
        # Δημιουργία γραφήματος
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Γράφημα παραμέτρων ανά επίπεδο
        ax1.bar(layers, parameters, color=self.colors['primary'], alpha=0.7)
        ax1.set_title('Παράμετροι ανά Επίπεδο', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Επίπεδο', fontsize=12)
        ax1.set_ylabel('Παράμετροι', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Γράφημα τύπων ενεργοποίησης
        if activations:
            activation_counts = {}
            for act in activations:
                activation_counts[act] = activation_counts.get(act, 0) + 1
            
            ax2.pie(activation_counts.values(), labels=activation_counts.keys(), autopct='%1.1f%%',
                    colors=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
            ax2.set_title('Τύποι Ενεργοποίησης', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Γράφημα αρχιτεκτονικής μοντέλου δημιουργήθηκε")
    
    def create_explainability_plots_from_data(self, explainability_data: dict):
        """
        Δημιουργία γραφημάτων explainability από πραγματικά δεδομένα.
        
        Args:
            explainability_data: Λεξικό με δεδομένα explainability
        """
        if not explainability_data:
            print("Δεν υπάρχουν δεδομένα explainability για τη δημιουργία γραφημάτων")
            return
        
        # Σύγκριση μεθόδων explainability
        methods_comparison = explainability_data.get('methods_comparison', {})
        if methods_comparison:
            methods = list(methods_comparison.keys())
            accuracy = [methods_comparison[m].get('accuracy', 0) for m in methods]
            interpretability = [methods_comparison[m].get('interpretability', 0) for m in methods]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Σύγκριση ακρίβειας
            bars1 = ax1.bar(methods, accuracy, color=self.colors['primary'], alpha=0.7)
            ax1.set_title('Ακρίβεια Εξηγήσεων', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Ακρίβεια', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars1, accuracy):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Σύγκριση ερμηνευσιμότητας
            bars2 = ax2.bar(methods, interpretability, color=self.colors['secondary'], alpha=0.7)
            ax2.set_title('Ερμηνευσιμότητα', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Ερμηνευσιμότητα', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars2, interpretability):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'explainability_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Σημαντικότητα χαρακτηριστικών
        feature_importance = explainability_data.get('feature_importance', {})
        if feature_importance:
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(features, importance, color=self.colors['success'], alpha=0.7)
            plt.title('Σημαντικότητα Χαρακτηριστικών', fontsize=16, fontweight='bold')
            plt.xlabel('Σημαντικότητα', fontsize=12)
            plt.ylabel('Χαρακτηριστικό', fontsize=12)
            
            for bar, value in zip(bars, importance):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.2f}', ha='left', va='center', fontweight='bold')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Γραφήματα explainability δημιουργήθηκαν από πραγματικά δεδομένα")
    
    def create_system_overview_plot(self):
        """
        Δημιουργία γραφήματος επισκόπησης συστήματος.
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Ορισμός στοιχείων συστήματος
        components = [
            ('Dataset\n(Cat/Dog)', 0.5, 0.9),
            ('Data\nPreprocessing', 0.5, 0.75),
            ('Model\n(ResNet18)', 0.5, 0.6),
            ('Training\nPipeline', 0.5, 0.45),
            ('Evaluation\nMetrics', 0.5, 0.3),
            ('Explainability\nTools', 0.5, 0.15)
        ]
        
        # Χρώματα για κάθε component
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'],
                 self.colors['success'], self.colors['dark'], self.colors['light']]
        
        # Σχεδίαση components
        for i, (component, x, y) in enumerate(components):
            rect = plt.Rectangle((x-0.15, y-0.05), 0.3, 0.1, 
                               facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, component, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Σχεδίαση βελών
        for i in range(len(components)-1):
            x1, y1 = components[i][1], components[i][2]
            x2, y2 = components[i+1][1], components[i+1][2]
            ax.annotate('', xy=(x2, y2+0.05), xytext=(x1, y1-0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Ρύθμιση άξονα
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('Επισκόπηση Συστήματος XAI', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'system_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Γράφημα επισκόπησης συστήματος δημιουργήθηκε")
    
    def generate_all_plots(self, data_dict: dict):
        """
        Δημιουργία όλων των γραφημάτων από πραγματικά δεδομένα.
        
        Args:
            data_dict: Λεξικό με όλα τα δεδομένα για τα γραφήματα
        """
        print("=" * 60)
        print("ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ ΑΠΟ ΠΡΑΓΜΑΤΙΚΑ ΔΕΔΟΜΕΝΑ")
        print("=" * 60)
        
        try:
            # Δημιουργία γραφημάτων εκπαίδευσης
            if 'training_data' in data_dict:
                self.create_training_plots_from_data(data_dict['training_data'])
            
            # Δημιουργία γραφημάτων απόδοσης
            if 'performance_data' in data_dict:
                self.create_performance_plots_from_data(data_dict['performance_data'])
            
            # Δημιουργία γραφημάτων dataset
            if 'dataset_info' in data_dict:
                self.create_dataset_plots_from_data(data_dict['dataset_info'])
            
            # Δημιουργία γραφήματος αρχιτεκτονικής
            if 'model_info' in data_dict:
                self.create_model_architecture_plot(data_dict['model_info'])
            
            # Δημιουργία γραφημάτων explainability
            if 'explainability_data' in data_dict:
                self.create_explainability_plots_from_data(data_dict['explainability_data'])
            
            # Δημιουργία γραφήματος επισκόπησης συστήματος
            self.create_system_overview_plot()
            
            print("\n" + "=" * 60)
            print("ΟΛΑ ΤΑ ΓΡΑΦΗΜΑΤΑ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ ΕΠΙΤΥΧΩΣ!")
            print("=" * 60)
            print(f"Φάκελος εξόδου: {self.output_dir}")
            
        except Exception as e:
            print(f"Σφάλμα στη δημιουργία γραφημάτων: {e}")


def create_plots_from_training_results(training_results: dict, output_dir: str = "outputs/plots"):
    """
    Συνάρτηση για τη δημιουργία γραφημάτων από αποτελέσματα εκπαίδευσης.
    
    Args:
        training_results: Λεξικό με τα αποτελέσματα εκπαίδευσης
        output_dir: Φάκελος εξόδου για τα γραφήματα
    """
    plot_generator = DynamicPlotGenerator(output_dir)
    
    # Δημιουργία γραφημάτων εκπαίδευσης
    if 'training_history' in training_results:
        plot_generator.create_training_plots_from_data(training_results['training_history'])
    
    # Δημιουργία γραφημάτων απόδοσης
    if 'evaluation_results' in training_results:
        plot_generator.create_performance_plots_from_data(training_results['evaluation_results'])
    
    print("Γραφήματα δημιουργήθηκαν επιτυχώς!")


def create_plots_from_evaluation_results(evaluation_results: dict, output_dir: str = "outputs/plots"):
    """
    Συνάρτηση για τη δημιουργία γραφημάτων από αποτελέσματα αξιολόγησης.
    
    Args:
        evaluation_results: Λεξικό με τα αποτελέσματα αξιολόγησης
        output_dir: Φάκελος εξόδου για τα γραφήματα
    """
    plot_generator = DynamicPlotGenerator(output_dir)
    plot_generator.create_performance_plots_from_data(evaluation_results)
    print("Γραφήματα αξιολόγησης δημιουργήθηκαν επιτυχώς!") 