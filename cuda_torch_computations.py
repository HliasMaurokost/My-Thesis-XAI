import torch
import time
import sys

# Συνάρτηση για έγχρωμη έξοδο (αν υποστηρίζεται)
def color_text(text, color="green"):
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

# Συνάρτηση για εκτύπωση τίτλων με διαχωριστικές γραμμές
def print_header(text):
    print(f"\n{'=' * 50}")
    print(f"|| {text}")
    print(f"{'=' * 50}")

# === Πληροφορίες για το σύστημα και την εγκατάσταση PyTorch ===
print_header("SYSTEM CONFIGURATION")
print(f"Έκδοση Python: {sys.version}")
print(f"Έκδοση PyTorch: {torch.__version__}")
print(f"Υποστήριξη CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

# === Ανίχνευση και εμφάνιση διαθέσιμων συσκευών GPU ===
print_header("GPU DETECTION")
gpu_count = torch.cuda.device_count()
print(f"Ανιχνεύθηκαν GPUs: {gpu_count} συσκευή/ες")

for i in range(gpu_count):
    print(f"\nΛεπτομέρειες GPU {i}:")
    print(f"  Όνομα: {torch.cuda.get_device_name(i)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    print(f"  Δεσμευμένη Μνήμη: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
    print(f"  Κρυφή Μνήμη (Cache): {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

# === Benchmark Υπολογιστικής Απόδοσης με Πολλαπλασιασμό Πινάκων ===
print_header("PERFORMANCE TEST")
size = 8192         # Πολύ μεγάλος πίνακας για σοβαρό υπολογισμό (~512MB/tensor)
iterations = 5      # Λιγότερες επαναλήψεις λόγω μεγέθους

def run_benchmark(device):
    x = torch.randn((size, size), dtype=torch.float32, device=device)
    y = torch.randn((size, size), dtype=torch.float32, device=device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(iterations):
        result = torch.matmul(x, y)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    return time.time() - start, result.cpu()

cpu_time, cpu_result = run_benchmark(torch.device("cpu"))
print(f"\nΧρόνος Εκτέλεσης σε CPU: {cpu_time:.4f} δευτερόλεπτα ({iterations} επαναλήψεις)")

# Εκτέλεση benchmark σε GPU
if torch.cuda.is_available():
    gpu_time, gpu_result = run_benchmark(torch.device("cuda"))
    speedup = cpu_time / gpu_time
    print(f"Χρόνος Εκτέλεσης σε GPU: {gpu_time:.4f} δευτερόλεπτα ({iterations} επαναλήψεις)")
    print(f"Συντελεστής Επιτάχυνσης (Speedup): {speedup:.2f}x")

    # === Ακριβής Έλεγχος Ορθότητας ===
    print_header("NUMERICAL ACCURACY COMPARISON")
    cpu_result = cpu_result.float()  # Βεβαιωνόμαστε ότι δεν είναι float64
    diff = torch.abs(cpu_result - gpu_result)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_error = (diff / (torch.abs(cpu_result) + 1e-8)).mean().item()

    print(f"Μέγιστη Απόκλιση: {max_diff:.6f}")
    print(f"Μέση Απόλυτη Απόκλιση: {mean_diff:.6f}")
    print(f"Μέσο Σχετικό Σφάλμα: {rel_error:.6f}")

    if torch.allclose(cpu_result, gpu_result, atol=1e-3, rtol=1e-2):
        print(color_text(" Τα αποτελέσματα είναι αποδεκτά εντός ορίων ανοχής (allclose OK)", "green"))
    else:
        print(color_text(" Τα αποτελέσματα έχουν αξιοσημείωτη διαφορά (allclose FAILED)", "red"))

else:
    print(color_text(" Δεν εντοπίστηκε GPU. Έγινε benchmarking μόνο στην CPU.", "yellow"))

# === Προχωρημένοι Έλεγχοι ===
print_header("ADVANCED CHECKS")
print(f"Υποστηρίζεται Mixed Precision (AMP): {torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7}")
print(f"cuDNN Ενεργό: {torch.backends.cudnn.enabled}")
print(f"cuDNN Fastest Mode: {torch.backends.cudnn.benchmark}")
print(f"XLA Accelerator (μόνο για TPU): False")

# Τέλος εκτέλεσης
print_header("VERIFICATION COMPLETE")
