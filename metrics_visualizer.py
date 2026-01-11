"""
Robust standalone script για οπτικοποίηση μετρικών εκπαίδευσης.

Αυτό το script διαβάζει το training_metrics_20250717_043607.txt και δημιουργεί
γραφήματα  για ανάλυση των αποτελεσμάτων.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ρύθμιση για ελληνικά γραφήματα
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class MetricsVisualizer:
    """
    Κλάση για δημιουργία γραφημάτων από μετρικές εκπαίδευσης.
    """
    
    def __init__(self, metrics_file_path: str):
        """
        Αρχικοποίηση του visualizer.
        
        Args:
            metrics_file_path: Διαδρομή προς το αρχείο μετρικών
        """
        self.metrics_file_path = Path(metrics_file_path)
        self.output_dir = self.metrics_file_path.parent / "metrics_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # vibrant χρώματα
        self.colors = {
            'primary': '#FF6B6B',      # Κοκκινό-ροζ
            'secondary': '#4ECDC4',    # Τυρκουάζ
            'accent': '#45B7D1',       # Μπλε
            'success': '#96CEB4',      # Πράσινο
            'warning': '#FFEAA7',      # Κίτρινο
            'info': '#DDA0DD',         # Μωβ
            'dark': '#2C3E50',         # Σκούρο μπλε
            'light': '#F8F9FA'         # Ανοιχτό γκρι
        }
        
        # Seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Φόρτωση δεδομένων
        self.data = self._parse_metrics_file()
        
    def _parse_metrics_file(self) -> dict:
        """
        Ανάλυση του αρχείου μετρικών.
        
        Returns:
            dict: Λεξικό με τα δεδομένα
        """
        try:
            with open(self.metrics_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {}
            
            # Εξαγωγή βασικών πληροφοριών
            data['training_date'] = re.search(r'Ημερομηνία εκπαίδευσης: (.+)', content).group(1)
            data['device'] = re.search(r'Device: (.+)', content).group(1)
            data['learning_rate'] = float(re.search(r'Learning rate: (.+)', content).group(1))
            data['batch_size'] = int(re.search(r'Batch size: (.+)', content).group(1))
            data['total_time'] = float(re.search(r'Συνολικός χρόνος εκπαίδευσης: (.+) δευτερόλεπτα', content).group(1))
            data['epochs'] = int(re.search(r'Εpochs που ολοκληρώθηκαν: (.+)', content).group(1))
            
            # Εξαγωγή μετρικών εκπαίδευσης
            epochs_data = []
            lines = content.split('\n')
            in_epochs_section = False
            
            for line in lines:
                if 'Epoch\tTrain Loss' in line:
                    in_epochs_section = True
                    continue
                elif in_epochs_section and line.strip() and not line.startswith('-'):
                    if line.strip() and not line.startswith('ΣΤΑΤΙΣΤΙΚΑ'):
                        # Καθαρισμός της γραμμής από επιπλέον κενά
                        clean_line = re.sub(r'\s+', '\t', line.strip())
                        parts = clean_line.split('\t')
                        if len(parts) >= 7:
                            try:
                                epochs_data.append({
                                    'epoch': int(parts[0]),
                                    'train_loss': float(parts[1]),
                                    'train_acc': float(parts[2]),
                                    'val_loss': float(parts[3]),
                                    'val_acc': float(parts[4]),
                                    'lr': float(parts[5]),
                                    'time': float(parts[6])
                                })
                            except ValueError as e:
                                print(f"Σφάλμα στην ανάλυση γραμμής: {line}")
                                continue
                elif line.startswith('ΣΤΑΤΙΣΤΙΚΑ'):
                    break
            
            data['epochs_data'] = epochs_data
            
            # Εξαγωγή στατιστικών
            data['best_val_loss'] = float(re.search(r'Καλύτερο validation loss: (.+) \(epoch', content).group(1))
            data['best_val_acc'] = float(re.search(r'Καλύτερη validation accuracy: (.+) \(epoch', content).group(1))
            data['avg_train_loss'] = float(re.search(r'Μέσος όρος train loss: (.+)', content).group(1))
            data['avg_val_loss'] = float(re.search(r'Μέσος όρος validation loss: (.+)', content).group(1))
            data['avg_train_acc'] = float(re.search(r'Μέσος όρος train accuracy: (.+)', content).group(1))
            data['avg_val_acc'] = float(re.search(r'Μέσος όρος validation accuracy: (.+)', content).group(1))
            
            # Εξαγωγή πληροφοριών μοντέλου
            data['total_params'] = int(re.search(r'Συνολικές παράμετροι: (.+)', content).group(1).replace(',', ''))
            data['model_size'] = float(re.search(r'Μέγεθος μοντέλου: (.+) MB', content).group(1))
            
            # Εξαγωγή πληροφοριών dataset
            data['train_samples'] = int(re.search(r'Δείγματα εκπαίδευσης: (.+)', content).group(1))
            data['val_samples'] = int(re.search(r'Δείγματα επικύρωσης: (.+)', content).group(1))
            data['test_samples'] = int(re.search(r'Δείγματα δοκιμής: (.+)', content).group(1))
            data['total_samples'] = int(re.search(r'Συνολικά δείγματα: (.+)', content).group(1))
            
            return data
            
        except Exception as e:
            print(f"Σφάλμα στην ανάλυση αρχείου: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def create_training_curves(self):
        """
        Δημιουργία γραφημάτων training curves.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress Analysis', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Loss curves
        ax1.plot(epochs, train_loss, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Loss', alpha=0.8)
        ax1.plot(epochs, val_loss, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Loss', alpha=0.8)
        ax1.set_title('Loss Curves', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Accuracy curves
        ax2.plot(epochs, train_acc, 'o-', color=self.colors['accent'], linewidth=3, 
                markersize=8, label='Training Accuracy', alpha=0.8)
        ax2.plot(epochs, val_acc, 's-', color=self.colors['success'], linewidth=3, 
                markersize=8, label='Validation Accuracy', alpha=0.8)
        ax2.set_title('Accuracy Curves', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(self.colors['light'])
        
        # Training time per epoch
        time_per_epoch = [d['time'] for d in epochs_data]
        ax3.bar(epochs, time_per_epoch, color=self.colors['warning'], alpha=0.8, 
               edgecolor=self.colors['dark'], linewidth=2)
        ax3.set_title('Training Time per Epoch', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(self.colors['light'])
        
        # Overfitting analysis
        overfitting = [t - v for t, v in zip(train_loss, val_loss)]
        ax4.plot(epochs, overfitting, 'o-', color=self.colors['info'], linewidth=3, 
                markersize=8, alpha=0.8)
        ax4.axhline(y=0, color=self.colors['dark'], linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Analysis (Train - Val Loss)', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Loss Difference', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor(self.colors['light'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Training curves δημιουργήθηκαν")
    
    def create_performance_summary(self):
        """
        Δημιουργία γραφήματος σύνοψης απόδοσης.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Summary', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Best metrics
        metrics = ['Best Val Accuracy', 'Best Val Loss', 'Avg Train Accuracy', 'Avg Val Accuracy']
        values = [self.data['best_val_acc'], self.data['best_val_loss'], 
                 self.data['avg_train_acc'], self.data['avg_val_acc']]
        colors = [self.colors['success'], self.colors['primary'], 
                 self.colors['accent'], self.colors['secondary']]
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, 
                      edgecolor=self.colors['dark'], linewidth=2)
        ax1.set_title('Key Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor(self.colors['light'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Dataset distribution
        labels = ['Training', 'Validation', 'Test']
        sizes = [self.data['train_samples'], self.data['val_samples'], self.data['test_samples']]
        colors_pie = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontweight': 'bold'})
        ax2.set_title('Dataset Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Model information
        model_info = ['Total Parameters', 'Model Size (MB)', 'Training Time (min)', 'Epochs']
        model_values = [self.data['total_params']/1e6, self.data['model_size'], 
                       self.data['total_time']/60, self.data['epochs']]
        model_labels = [f'{v:.1f}M' if 'Parameters' in info else f'{v:.1f}' 
                       for info, v in zip(model_info, model_values)]
        
        bars = ax3.bar(model_info, model_values, color=[self.colors['info'], self.colors['warning'], 
                                                       self.colors['success'], self.colors['accent']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax3.set_title('Model & Training Information', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, label in zip(bars, model_labels):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Training efficiency
        efficiency_metrics = ['Samples/sec', 'Epochs/min', 'Accuracy/Time']
        efficiency_values = [
            self.data['total_samples'] / self.data['total_time'],
            self.data['epochs'] / (self.data['total_time'] / 60),
            self.data['best_val_acc'] / (self.data['total_time'] / 60)
        ]
        
        bars = ax4.bar(efficiency_metrics, efficiency_values, color=[self.colors['primary'], 
                                                                    self.colors['secondary'], 
                                                                    self.colors['success']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax4.set_title('Training Efficiency', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Rate', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, efficiency_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Performance summary δημιουργήθηκε")
    
    def create_loss_analysis(self):
        """
        Δημιουργία λεπτομερούς ανάλυσης loss.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Loss Analysis & Convergence', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Loss comparison with confidence bands
        ax1.plot(epochs, train_loss, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Loss', alpha=0.8)
        ax1.plot(epochs, val_loss, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Loss', alpha=0.8)
        
        # Add confidence bands
        train_std = np.std(train_loss)
        val_std = np.std(val_loss)
        ax1.fill_between(epochs, np.array(train_loss) - train_std, np.array(train_loss) + train_std,
                        alpha=0.2, color=self.colors['primary'])
        ax1.fill_between(epochs, np.array(val_loss) - val_std, np.array(val_loss) + val_std,
                        alpha=0.2, color=self.colors['secondary'])
        
        ax1.set_title('Loss Curves with Confidence Bands', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Loss improvement analysis
        train_improvement = [0] + [train_loss[i] - train_loss[i-1] for i in range(1, len(train_loss))]
        val_improvement = [0] + [val_loss[i] - val_loss[i-1] for i in range(1, len(val_loss))]
        
        x = np.arange(len(epochs))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, train_improvement, width, label='Training Loss Change', 
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, val_improvement, width, label='Validation Loss Change', 
                       color=self.colors['secondary'], alpha=0.8)
        
        ax2.set_title('Loss Improvement per Epoch', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss Change', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(epochs)
        ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        ax2.axhline(y=0, color=self.colors['dark'], linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Loss analysis δημιουργήθηκε")
    
    def create_accuracy_heatmap(self):
        """
        Δημιουργία heatmap για accuracy analysis.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        
        # Δημιουργία epochs labels
        epochs = [f'Epoch {d["epoch"]}' for d in epochs_data]
        
        
        # Δημιουργία heatmap με διαφορετικά ranges για accuracy και loss
        # Accuracy: 0.7-1.0 (πράσινο = καλό)
        # Loss: 0.0-0.3 (κόκκινο = κακό, πράσινο = καλό)
        
        # Δημιουργία δύο ξεχωριστών heatmaps
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy heatmap (πάνω)
        acc_matrix = np.array([
            [d['train_acc'] for d in epochs_data],
            [d['val_acc'] for d in epochs_data]
        ])
        
        im1 = ax1.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
        ax1.set_xticks(range(len(epochs)))
        ax1.set_yticks(range(2))
        ax1.set_xticklabels(epochs, fontweight='bold')
        ax1.set_yticklabels(['Train Acc', 'Val Acc'], fontweight='bold')
        ax1.set_title('Accuracy Metrics', fontsize=16, fontweight='bold', pad=15)
        
        # Προσθήκη τιμών στα κελιά για accuracy
        for i in range(2):
            for j in range(len(epochs)):
                text = ax1.text(j, i, f'{acc_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Loss heatmap (κάτω)
        loss_matrix = np.array([
            [d['train_loss'] for d in epochs_data],
            [d['val_loss'] for d in epochs_data]
        ])
        
        im2 = ax2.imshow(loss_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=0.3)
        ax2.set_xticks(range(len(epochs)))
        ax2.set_yticks(range(2))
        ax2.set_xticklabels(epochs, fontweight='bold')
        ax2.set_yticklabels(['Train Loss', 'Val Loss'], fontweight='bold')
        ax2.set_title('Loss Metrics (Lower is Better)', fontsize=16, fontweight='bold', pad=15)
        
        # Προσθήκη τιμών στα κελιά για loss
        for i in range(2):
            for j in range(len(epochs)):
                text = ax2.text(j, i, f'{loss_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Colorbars
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Accuracy Score', fontweight='bold')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Loss Value', fontweight='bold')
        
        # Γενικός τίτλος
        fig.suptitle('Training Metrics Heatmap', fontsize=20, fontweight='bold', 
                    color=self.colors['dark'], y=0.95)
        
        ax1.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Accuracy heatmap δημιουργήθηκε")
    
    def create_training_efficiency(self):
        """
        Δημιουργία γραφήματος training efficiency.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        time_per_epoch = [d['time'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Efficiency Analysis', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Time vs Accuracy scatter
        scatter = ax1.scatter(time_per_epoch, train_acc, c=epochs, cmap='viridis', 
                             s=200, alpha=0.8, edgecolors=self.colors['dark'], linewidth=2)
        ax1.scatter(time_per_epoch, val_acc, c=epochs, cmap='viridis', 
                   s=200, alpha=0.6, edgecolors=self.colors['dark'], linewidth=2, marker='s')
        
        ax1.set_title('Training Time vs Accuracy', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Time per Epoch (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Colorbar for epochs
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Epoch', fontweight='bold')
        
        # Efficiency metrics over time
        efficiency = [acc/time for acc, time in zip(train_acc, time_per_epoch)]
        val_efficiency = [acc/time for acc, time in zip(val_acc, time_per_epoch)]
        
        ax2.plot(epochs, efficiency, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Efficiency (Acc/Time)', alpha=0.8)
        ax2.plot(epochs, val_efficiency, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Efficiency (Acc/Time)', alpha=0.8)
        
        ax2.set_title('Efficiency Over Time', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Efficiency (Accuracy/Time)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(self.colors['light'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_efficiency.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Training efficiency δημιουργήθηκε")
    
    def create_model_comparison(self):
        """
        Δημιουργία γραφήματος σύγκρισης μοντέλου με benchmarks.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance Analysis', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Αληθινά δεδομένα μοντέλου
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        
        # Accuracy progression
        ax1.plot(epochs, train_acc, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Accuracy', alpha=0.8)
        ax1.plot(epochs, val_acc, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Accuracy', alpha=0.8)
        ax1.set_title('Accuracy Progression', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Model metrics analysis
        metrics = ['Best Val Acc', 'Avg Train Acc', 'Best Val Loss', 'Avg Train Loss']
        values = [self.data['best_val_acc'], self.data['avg_train_acc'], 
                 self.data['best_val_loss'], self.data['avg_train_loss']]  # Πραγματικές τιμές loss
        colors = [self.colors['success'], self.colors['primary'], 
                 self.colors['secondary'], self.colors['accent']]
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8, 
                      edgecolor=self.colors['dark'], linewidth=2)
        ax2.set_title('Key Model Metrics', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Model analysis δημιουργήθηκε")
    
    def create_convergence_analysis(self):
        """
        Δημιουργία γραφήματος ανάλυσης σύγκλισης.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Convergence Analysis & Learning Dynamics', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Learning rate analysis
        lr_values = [d['lr'] for d in epochs_data]
        ax1.plot(epochs, lr_values, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, alpha=0.8)
        ax1.set_title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Convergence speed analysis
        train_loss_improvement = [0] + [train_loss[i] - train_loss[i-1] for i in range(1, len(train_loss))]
        val_loss_improvement = [0] + [val_loss[i] - val_loss[i-1] for i in range(1, len(val_loss))]
        
        x = np.arange(len(epochs))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, train_loss_improvement, width, label='Training Loss Change', 
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, val_loss_improvement, width, label='Validation Loss Change', 
                       color=self.colors['secondary'], alpha=0.8)
        
        ax2.set_title('Convergence Speed Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss Change', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(epochs)
        ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        ax2.axhline(y=0, color=self.colors['dark'], linestyle='-', alpha=0.5)
        
        # Generalization gap analysis
        generalization_gap = [t - v for t, v in zip(train_acc, val_acc)]
        ax3.plot(epochs, generalization_gap, 'o-', color=self.colors['accent'], linewidth=3, 
                markersize=8, alpha=0.8)
        ax3.fill_between(epochs, generalization_gap, alpha=0.3, color=self.colors['accent'])
        ax3.set_title('Generalization Gap Analysis', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Train Acc - Val Acc', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor(self.colors['light'])
        ax3.axhline(y=0, color=self.colors['dark'], linestyle='--', alpha=0.5)
        
        # Training momentum analysis
        momentum_metrics = ['Loss Momentum', 'Accuracy Momentum', 'Stability Index']
        momentum_values = [
            abs(train_loss_improvement[-1]) / abs(train_loss_improvement[1]) if train_loss_improvement[1] != 0 else 0,
            abs(train_acc[-1] - train_acc[0]) / (epochs[-1] - epochs[0]),
            np.std(val_acc) / np.mean(val_acc) if np.mean(val_acc) != 0 else 0
        ]
        
        bars = ax4.bar(momentum_metrics, momentum_values, color=[self.colors['success'], self.colors['warning'], self.colors['info']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax4.set_title('Training Momentum & Stability', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Momentum Score', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, momentum_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Convergence analysis δημιουργήθηκε")
    
    def create_dataset_insights(self):
        """
        Δημιουργία γραφήματος insights για το dataset.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Insights & Model Capacity Analysis', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Dataset size analysis
        dataset_sizes = [self.data['train_samples'], self.data['val_samples'], self.data['test_samples']]
        dataset_labels = ['Training', 'Validation', 'Test']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        wedges, texts, autotexts = ax1.pie(dataset_sizes, labels=dataset_labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontweight': 'bold'})
        ax1.set_title('Dataset Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Model capacity analysis
        capacity_metrics = ['Parameters (M)', 'Model Size (MB)', 'Samples/Param', 'Efficiency Ratio']
        capacity_values = [
            self.data['total_params'] / 1e6,
            self.data['model_size'],
            self.data['total_samples'] / self.data['total_params'],
            self.data['best_val_acc'] / (self.data['total_params'] / 1e6)
        ]
        
        bars = ax2.bar(capacity_metrics, capacity_values, color=[self.colors['success'], self.colors['primary'], 
                                                               self.colors['secondary'], self.colors['accent']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax2.set_title('Model Capacity Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, capacity_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training efficiency by dataset size
        efficiency_metrics = ['Samples/sec', 'Epochs/min', 'Accuracy/Time', 'Data Utilization']
        efficiency_values = [
            self.data['total_samples'] / self.data['total_time'],
            self.data['epochs'] / (self.data['total_time'] / 60),
            self.data['best_val_acc'] / (self.data['total_time'] / 60),
            (self.data['train_samples'] / self.data['total_samples']) * 100
        ]
        
        bars = ax3.bar(efficiency_metrics, efficiency_values, color=[self.colors['info'], self.colors['warning'], 
                                                                   self.colors['success'], self.colors['primary']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax3.set_title('Training Efficiency Metrics', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, efficiency_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Model performance vs dataset characteristics
        performance_metrics = ['Best Accuracy', 'Avg Accuracy', 'Loss Reduction', 'Convergence Speed']
        performance_values = [
            self.data['best_val_acc'],
            self.data['avg_val_acc'],
            1 - (self.data['avg_val_loss'] / self.data['avg_train_loss']),
            self.data['epochs'] / self.data['total_time']
        ]
        
        bars = ax4.bar(performance_metrics, performance_values, color=[self.colors['success'], self.colors['primary'], 
                                                                     self.colors['secondary'], self.colors['accent']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax4.set_title('Performance vs Dataset Characteristics', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_insights.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Dataset insights δημιουργήθηκε")
    
    def create_learning_dynamics(self):
        """
        Δημιουργία γραφήματος learning dynamics.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        time_per_epoch = [d['time'] for d in epochs_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Dynamics & Training Behavior', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Learning velocity analysis
        train_velocity = [0] + [abs(train_acc[i] - train_acc[i-1]) for i in range(1, len(train_acc))]
        val_velocity = [0] + [abs(val_acc[i] - val_acc[i-1]) for i in range(1, len(val_acc))]
        
        ax1.plot(epochs, train_velocity, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Velocity', alpha=0.8)
        ax1.plot(epochs, val_velocity, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Velocity', alpha=0.8)
        ax1.set_title('Learning Velocity Analysis', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy Change', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Training consistency analysis
        consistency_metrics = ['Train Loss Std', 'Val Loss Std', 'Train Acc Std', 'Val Acc Std']
        consistency_values = [
            np.std(train_loss),
            np.std(val_loss),
            np.std(train_acc),
            np.std(val_acc)
        ]
        
        bars = ax2.bar(consistency_metrics, consistency_values, color=[self.colors['accent'], self.colors['success'], 
                                                                     self.colors['primary'], self.colors['secondary']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax2.set_title('Training Consistency Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, consistency_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Time efficiency analysis
        time_efficiency = [acc/time for acc, time in zip(train_acc, time_per_epoch)]
        val_time_efficiency = [acc/time for acc, time in zip(val_acc, time_per_epoch)]
        
        ax3.plot(epochs, time_efficiency, 'o-', color=self.colors['warning'], linewidth=3, 
                markersize=8, label='Training Time Efficiency', alpha=0.8)
        ax3.plot(epochs, val_time_efficiency, 's-', color=self.colors['info'], linewidth=3, 
                markersize=8, label='Validation Time Efficiency', alpha=0.8)
        ax3.set_title('Time Efficiency Analysis', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy per Second', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor(self.colors['light'])
        
        # Learning curve characteristics
        curve_metrics = ['Learning Rate', 'Convergence Speed', 'Stability Index', 'Generalization Gap']
        curve_values = [
            self.data['learning_rate'],
            (train_acc[-1] - train_acc[0]) / (epochs[-1] - epochs[0]),
            np.std(val_acc) / np.mean(val_acc) if np.mean(val_acc) != 0 else 0,
            np.mean([t - v for t, v in zip(train_acc, val_acc)])
        ]
        
        bars = ax4.bar(curve_metrics, curve_values, color=[self.colors['primary'], self.colors['secondary'], 
                                                         self.colors['accent'], self.colors['success']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax4.set_title('Learning Curve Characteristics', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Characteristic Value', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, curve_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_dynamics.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Learning dynamics δημιουργήθηκε")
    
    def create_performance_radar(self):
        """
        Δημιουργία radar chart για performance analysis.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Performance Radar Analysis', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Performance metrics radar
        metrics = ['Best Val Acc', 'Avg Train Acc', 'Loss Reduction', 'Training Speed', 'Model Efficiency', 'Stability']
        values = [
            self.data['best_val_acc'],
            self.data['avg_train_acc'],
            1 - (self.data['avg_val_loss'] / self.data['avg_train_loss']),
            self.data['total_samples'] / self.data['total_time'],
            self.data['best_val_acc'] / (self.data['total_params'] / 1e6),
            1 - (np.std(val_acc) / np.mean(val_acc)) if np.mean(val_acc) != 0 else 0
        ]
        
        # Normalize values to 0-1 scale
        normalized_values = [v / max(values) if max(values) > 0 else 0 for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1.plot(angles, normalized_values, 'o-', linewidth=3, color=self.colors['primary'], alpha=0.8)
        ax1.fill(angles, normalized_values, alpha=0.25, color=self.colors['primary'])
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics, fontsize=10, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Performance Radar', fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Training progression radar
        epoch_metrics = ['Epoch 1', 'Epoch 2', 'Epoch 3']
        epoch_values = [val_acc[0], val_acc[1], val_acc[2]]
        
        # Normalize epoch values
        normalized_epoch_values = [v / max(epoch_values) if max(epoch_values) > 0 else 0 for v in epoch_values]
        
        angles_epoch = np.linspace(0, 2 * np.pi, len(epoch_metrics), endpoint=False).tolist()
        normalized_epoch_values += normalized_epoch_values[:1]
        angles_epoch += angles_epoch[:1]
        
        ax2.plot(angles_epoch, normalized_epoch_values, 'o-', linewidth=3, color=self.colors['secondary'], alpha=0.8)
        ax2.fill(angles_epoch, normalized_epoch_values, alpha=0.25, color=self.colors['secondary'])
        ax2.set_xticks(angles_epoch[:-1])
        ax2.set_xticklabels(epoch_metrics, fontsize=10, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.set_title('Epoch Progression Radar', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(self.colors['light'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Performance radar δημιουργήθηκε")
    
    def create_training_summary(self):
        """
        Δημιουργία comprehensive training summary.
        """
        if not self.data.get('epochs_data'):
            return
            
        epochs_data = self.data['epochs_data']
        epochs = [d['epoch'] for d in epochs_data]
        train_acc = [d['train_acc'] for d in epochs_data]
        val_acc = [d['val_acc'] for d in epochs_data]
        train_loss = [d['train_loss'] for d in epochs_data]
        val_loss = [d['val_loss'] for d in epochs_data]
        time_per_epoch = [d['time'] for d in epochs_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Training Summary', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Training progress overview
        ax1.plot(epochs, train_acc, 'o-', color=self.colors['primary'], linewidth=3, 
                markersize=8, label='Training Accuracy', alpha=0.8)
        ax1.plot(epochs, val_acc, 's-', color=self.colors['secondary'], linewidth=3, 
                markersize=8, label='Validation Accuracy', alpha=0.8)
        ax1.plot(epochs, train_loss, '^-', color=self.colors['accent'], linewidth=3, 
                markersize=8, label='Training Loss', alpha=0.8)
        ax1.plot(epochs, val_loss, 'v-', color=self.colors['success'], linewidth=3, 
                markersize=8, label='Validation Loss', alpha=0.8)
        ax1.set_title('Training Progress Overview', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(self.colors['light'])
        
        # Performance metrics comparison
        performance_metrics = ['Best Val Acc', 'Avg Train Acc', 'Best Val Loss', 'Avg Val Loss', 'Training Time', 'Model Size']
        performance_values = [
            self.data['best_val_acc'],
            self.data['avg_train_acc'],
            self.data['best_val_loss'],
            self.data['avg_val_loss'],
            self.data['total_time'] / 60,  # Convert to minutes
            self.data['model_size']
        ]
        
        # Normalize values for comparison
        normalized_values = [v / max(performance_values) if max(performance_values) > 0 else 0 for v in performance_values]
        
        bars = ax2.bar(performance_metrics, normalized_values, color=[self.colors['success'], self.colors['primary'], 
                                                                    self.colors['secondary'], self.colors['accent'], 
                                                                    self.colors['warning'], self.colors['info']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax2.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, performance_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training efficiency analysis
        efficiency_metrics = ['Samples/sec', 'Epochs/min', 'Accuracy/Time', 'Loss/Time', 'Efficiency Index']
        efficiency_values = [
            self.data['total_samples'] / self.data['total_time'],
            self.data['epochs'] / (self.data['total_time'] / 60),
            self.data['best_val_acc'] / (self.data['total_time'] / 60),
            self.data['avg_val_loss'] / (self.data['total_time'] / 60),
            (self.data['best_val_acc'] * self.data['total_samples']) / (self.data['total_time'] * self.data['total_params'] / 1e6)
        ]
        
        bars = ax3.bar(efficiency_metrics, efficiency_values, color=[self.colors['info'], self.colors['warning'], 
                                                                   self.colors['success'], self.colors['primary'], 
                                                                   self.colors['secondary']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax3.set_title('Training Efficiency Analysis', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, efficiency_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training quality assessment
        quality_metrics = ['Accuracy Quality', 'Loss Quality', 'Stability Quality', 'Efficiency Quality', 'Overall Quality']
        quality_values = [
            self.data['best_val_acc'],
            1 - self.data['avg_val_loss'],
            1 - (np.std(val_acc) / np.mean(val_acc)) if np.mean(val_acc) != 0 else 0,
            (self.data['total_samples'] / self.data['total_time']) / 100,  # Normalize
            (self.data['best_val_acc'] + (1 - self.data['avg_val_loss']) + 
             (1 - (np.std(val_acc) / np.mean(val_acc)) if np.mean(val_acc) != 0 else 0)) / 3
        ]
        
        bars = ax4.bar(quality_metrics, quality_values, color=[self.colors['success'], self.colors['primary'], 
                                                             self.colors['secondary'], self.colors['accent'], 
                                                             self.colors['warning']], 
                      alpha=0.8, edgecolor=self.colors['dark'], linewidth=2)
        ax4.set_title('Training Quality Assessment', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(self.colors['light'])
        
        # Add value labels
        for bar, value in zip(bars, quality_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_summary.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("✓ Training summary δημιουργήθηκε")
    
    def generate_all_plots(self):
        """
        Δημιουργία όλων των γραφημάτων.
        """
        print("=" * 60)
        print("ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ ΜΕΤΡΙΚΩΝ ΕΚΠΑΙΔΕΥΣΗΣ")
        print("=" * 60)
        print(f"Αρχείο: {self.metrics_file_path}")
        print(f"Έξοδος: {self.output_dir}")
        print("=" * 60)
        
        try:
            self.create_training_curves()
            self.create_performance_summary()
            self.create_loss_analysis()
            self.create_accuracy_heatmap()
            self.create_training_efficiency()
            self.create_model_comparison()
            self.create_convergence_analysis()
            self.create_dataset_insights()
            self.create_learning_dynamics()
            self.create_performance_radar()
            self.create_training_summary()
            
            print("=" * 60)
            print("ΟΛΑ ΤΑ ΓΡΑΦΗΜΑΤΑ ΔΗΜΙΟΥΡΓΗΘΗΚΑΝ ΕΠΙΤΥΧΩΣ!")
            print("=" * 60)
            print(f"Αποθηκεύτηκαν στο: {self.output_dir}")
            print("Δημιουργήθηκαν 11 γραφήματα:")
            print("1. training_curves.png - Training progress")
            print("2. performance_summary.png - Performance overview")
            print("3. loss_analysis.png - Loss analysis")
            print("4. accuracy_heatmap.png - Metrics heatmap")
            print("5. training_efficiency.png - Efficiency analysis")
            print("6. model_analysis.png - Model analysis")
            print("7. convergence_analysis.png - Convergence analysis")
            print("8. dataset_insights.png - Dataset insights")
            print("9. learning_dynamics.png - Learning dynamics")
            print("10. performance_radar.png - Performance radar")
            print("11. training_summary.png - Training summary")
            print("=" * 60)
            
        except Exception as e:
            print(f"Σφάλμα στη δημιουργία γραφημάτων: {e}")


def main():
    """
    Κύρια συνάρτηση.
    """
    # Αυτόματη εύρεση του αρχείου μετρικών
    metrics_file = "outputs/metrics/training_metrics_20250909_040714.txt"
    
    if not os.path.exists(metrics_file):
        print(f"Δεν βρέθηκε το αρχείο: {metrics_file}")
        print("Παρακαλώ βεβαιωθείτε ότι το αρχείο υπάρχει στη σωστή διαδρομή.")
        return
    
    # Δημιουργία visualizer
    visualizer = MetricsVisualizer(metrics_file)
    
    # Δημιουργία όλων των γραφημάτων
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
