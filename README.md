Οδηγός και Δομή Εφαρμογής XAI - Ταξινόμηση Εικόνων Γάτας/Σκύλου

Εισαγωγή

Αυτό το έγγραφο παρέχει μια αναλυτική περιγραφή της δομής και λειτουργίας της εφαρμογής XAI για την ταξινόμηση εικόνων γάτας και σκύλου. Η εφαρμογή υλοποιεί βαθιά μάθηση με τεχνικές ερμηνεύσιμης μάθησης (Explainable AI) για να εξηγεί τις αποφάσεις του μοντέλου.

APP/
├── main.py                     Κύρια είσοδος
├── utils/                      Βοηθητικές συναρτήσεις
│   ├── config.py              Ρυθμίσεις και configuration
│   └── data_utils.py          Χειρισμός dataset
├── models/                     Αρχιτεκτονικές μοντέλων
│   └── cnn_model.py           CNN μοντέλα
├── training/                   Εκπαίδευση μοντέλων
│   └── trainer.py             Κύρια λογική εκπαίδευσης
├── explainability/             XAI τεχνικές
│   ├── explainer_factory.py   Factory pattern για explainers
│   ├── gradcam_explainer.py   Grad-CAM implementation
│   ├── lime_explainer.py      LIME implementation
│   ├── shap_explainer.py      SHAP implementation
│   └── rule_based_explainer.py  Rule-based explanations
├── Dataset/                    Αρχικό dataset
│   ├── Cat/                   Εικόνες γάτας
│   └── Dog/                   Εικόνες σκύλου
├── outputs/                    Αποτελέσματα
│   ├── models/                Αποθηκευμένα μοντέλα
│   ├── plots/                 Γραφήματα
│   ├── metrics/               Μετρικές
│   ├── gradcam/               Grad-CAM αποτελέσματα
│   ├── lime/                  LIME αποτελέσματα
│   ├── shap/                  SHAP αποτελέσματα
│   └── xai_info/              Πληροφορίες XAI
└── plot_generator.py          Δημιουργός γραφημάτων
