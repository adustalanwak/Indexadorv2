#!/usr/bin/env python3
"""
Indexador de PDFs y Chat con Ollama
Aplicación de escritorio para indexar PDFs y consultar información contable usando Ollama.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui import main

if __name__ == '__main__':
    main()
