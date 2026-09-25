import sys
from pathlib import Path

# __file__ è in postprocessing/, parents[1] risale alla radice del progetto
project_root = str(Path(__file__).resolve().parents[1])

# Usa append per non sovrascrivere le librerie standard di Python
# Il controllo if evita di duplicare il percorso nel sys.path
if project_root not in sys.path:
    sys.path.append(project_root)