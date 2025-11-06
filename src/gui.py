from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QTextEdit, QLabel, QFileDialog,
    QProgressBar, QMessageBox, QSplitter, QComboBox, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import os
import sys
from db import Database
from pdf_processor import PDFProcessor
from ollama_client import OllamaClient
from rag_system import RAGSystem

class IndexingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, folder_path, db, processor):
        super().__init__()
        self.folder_path = folder_path
        self.db = db
        self.processor = processor

    def run(self):
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
            total = len(pdf_files)
            for i, filename in enumerate(pdf_files):
                filepath = os.path.join(self.folder_path, filename)
                text = self.processor.extract_text_from_pdf(filepath)
                tables_result = self.processor.extract_tables_hybrid(filepath)
                tables = tables_result.get('tables', [])
                metadata = self.processor.extract_document_data(text, tables)

                # Index content in Ollama knowledge base for learning
                if hasattr(self, 'ollama_client'):
                    self.ollama_client.index_pdf_content(text, metadata)

                # Index content in RAG system for semantic search
                if hasattr(self, 'rag_system'):
                    self.rag_system.add_pdf_document(filename, text, metadata)

                self.db.insert_pdf(filename, filepath, text, metadata)
                self.progress.emit(int((i + 1) / total * 100))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.processor = PDFProcessor()
        self.ollama = OllamaClient()
        self.rag_system = RAGSystem()  # Initialize RAG system
        self.current_pdf_id = None
        self.init_ui()

    def on_model_changed(self):
        """Update Ollama client when model changes"""
        selected_model = self.model_combo.currentText()
        self.ollama = OllamaClient(model=selected_model)
        self.chat_history.append(f"Modelo cambiado a: {selected_model}\n\n")

    def init_ui(self):
        self.setWindowTitle('Indexador de PDFs y Chat con Ollama')
        self.setGeometry(100, 100, 1000, 700)

        # Set up light theme with better contrast
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            QPushButton {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QListWidget {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                alternate-background-color: #f9f9f9;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
            }
            QLabel {
                color: #333333;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QComboBox {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel - PDF management
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Ollama Configuration Group
        config_group = QGroupBox('Configuración de Ollama')
        config_layout = QVBoxLayout(config_group)

        # Server info (read-only)
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel('Servidor:'))
        server_label = QLabel('http://10.65.117.238:11434')
        server_label.setStyleSheet("font-weight: bold; color: #006400;")
        server_layout.addWidget(server_label)
        server_layout.addStretch()
        config_layout.addLayout(server_layout)

        # Model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('Modelo:'))
        self.model_combo = QComboBox()
        # Get available models from Ollama server
        available_models = self.ollama.get_available_models_list()
        if available_models:
            self.model_combo.addItems(available_models)
            # Try to set default model if available
            if 'granite4:tiny-h' in available_models:
                self.model_combo.setCurrentText('granite4:tiny-h')
            else:
                self.model_combo.setCurrentIndex(0)  # Select first available model
        else:
            # Fallback to hardcoded list if server is not available
            self.model_combo.addItems([
                'granite4:tiny-h',
                'llama2',
                'mistral',
                'codellama',
                'vicuna',
                'orca-mini',
                'phi'
            ])
            self.model_combo.setCurrentText('granite4:tiny-h')
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)

        left_layout.addWidget(config_group)

        self.select_folder_btn = QPushButton('Seleccionar Carpeta')
        self.select_folder_btn.clicked.connect(self.select_folder)
        left_layout.addWidget(self.select_folder_btn)

        self.index_btn = QPushButton('Indexar PDFs')
        self.index_btn.clicked.connect(self.index_pdfs)
        self.index_btn.setEnabled(False)
        left_layout.addWidget(self.index_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.pdf_list = QListWidget()
        self.pdf_list.itemClicked.connect(self.on_pdf_selected)
        left_layout.addWidget(QLabel('PDFs Indexados:'))
        left_layout.addWidget(self.pdf_list)

        # Clear buttons
        self.clear_selection_btn = QPushButton('Limpiar Selección')
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        left_layout.addWidget(self.clear_selection_btn)

        self.clear_db_btn = QPushButton('Limpiar Base de Datos')
        self.clear_db_btn.clicked.connect(self.clear_database)
        self.clear_db_btn.setStyleSheet("QPushButton { background-color: #8B0000; color: #ffffff; } QPushButton:hover { background-color: #A00000; }")
        left_layout.addWidget(self.clear_db_btn)

        # Right panel - Chat
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        right_layout.addWidget(QLabel('Historial de Chat:'))
        right_layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(60)
        self.chat_input.setPlaceholderText('Escribe tu pregunta sobre los PDFs...')
        input_layout.addWidget(self.chat_input)

        self.send_btn = QPushButton('Enviar')
        self.send_btn.clicked.connect(self.send_query)
        input_layout.addWidget(self.send_btn)

        right_layout.addLayout(input_layout)

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

        self.load_indexed_pdfs()
        self.load_existing_pdfs_to_rag()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Seleccionar Carpeta con PDFs')
        if folder:
            self.selected_folder = folder
            self.index_btn.setEnabled(True)
            self.select_folder_btn.setText(f'Carpeta: {os.path.basename(folder)}')

    def index_pdfs(self):
        if not hasattr(self, 'selected_folder'):
            return

        self.progress_bar.setVisible(True)
        self.index_btn.setEnabled(False)

        self.indexing_thread = IndexingThread(self.selected_folder, self.db, self.processor)
        self.indexing_thread.ollama_client = self.ollama  # Pass Ollama client for indexing
        self.indexing_thread.rag_system = self.rag_system  # Pass RAG system for indexing
        self.indexing_thread.progress.connect(self.progress_bar.setValue)
        self.indexing_thread.finished.connect(self.on_indexing_finished)
        self.indexing_thread.error.connect(self.on_indexing_error)
        self.indexing_thread.start()

        # Also load existing PDFs into RAG system if any
        self.load_existing_pdfs_to_rag()

    def on_indexing_finished(self):
        self.progress_bar.setVisible(False)
        self.index_btn.setEnabled(True)
        self.load_indexed_pdfs()
        indexed_count = self.rag_system.get_document_count()
        QMessageBox.information(self, 'Éxito', f'PDFs indexados correctamente.\nDocumentos en base de conocimiento RAG: {indexed_count}')

    def on_indexing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.index_btn.setEnabled(True)
        QMessageBox.critical(self, 'Error', f'Error al indexar PDFs: {error_msg}')

    def load_indexed_pdfs(self):
        self.pdf_list.clear()
        pdfs = self.db.get_all_pdfs()
        for pdf_id, filename, filepath in pdfs:
            self.pdf_list.addItem(f"{pdf_id}: {filename}")

    def on_pdf_selected(self, item):
        pdf_id = int(item.text().split(':')[0])
        self.current_pdf_id = pdf_id
        text = self.db.get_pdf_text(pdf_id)
        if text:
            # Extract structured data
            tables_result = self.processor.extract_tables_hybrid(self.db.get_all_pdfs()[pdf_id-1][2])  # filepath
            tables = tables_result.get('tables', [])
            metadata = self.processor.extract_document_data(text, tables)
            self.chat_history.append(f"PDF seleccionado: {item.text()}\n")
            if metadata:
                import json
                data = json.loads(metadata)
                self.chat_history.append("Datos extraídos:\n")
                for key, value in data.items():
                    if key != 'tablas':
                        self.chat_history.append(f"{key.capitalize()}: {value}\n")
                if 'tablas' in data:
                    self.chat_history.append(f"Tablas encontradas: {len(data['tablas'])}\n")
            self.chat_history.append(f"Texto extraído:\n{text[:500]}...\n\n")

    def send_query(self):
        query = self.chat_input.toPlainText().strip()
        if not query:
            return

        self.chat_history.append(f"Tú: {query}\n")
        self.chat_input.clear()

        # Use RAG system for semantic search and retrieval
        response = self.rag_system.query(query)
        self.chat_history.append(f"Ollama (RAG): {response}\n\n")
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def clear_selection(self):
        self.current_pdf_id = None
        self.pdf_list.clearSelection()
        self.chat_history.append("Selección limpiada.\n\n")

    def load_existing_pdfs_to_rag(self):
        """Load all existing PDFs from database into RAG system asynchronously"""
        try:
            pdfs = self.db.get_all_pdfs()
            if not pdfs:
                return  # No PDFs to load

            # Show loading message
            self.chat_history.append("Cargando PDFs existentes al sistema RAG...\n")

            # Process in small batches to avoid blocking UI
            batch_size = 5
            for i in range(0, len(pdfs), batch_size):
                batch = pdfs[i:i + batch_size]
                for pdf_id, filename, filepath in batch:
                    text = self.db.get_pdf_text(pdf_id)
                    if text:
                        # Get metadata if available
                        metadata = self.db.get_pdf_metadata(pdf_id)
                        self.rag_system.add_pdf_document(filename, text, metadata)

                # Allow UI to update between batches
                QApplication.processEvents()

            loaded_count = self.rag_system.get_document_count()
            self.chat_history.append(f"PDFs cargados al sistema RAG: {loaded_count} documentos\n\n")

        except KeyboardInterrupt:
            # Handle user interruption gracefully
            self.chat_history.append("Carga de PDFs interrumpida por el usuario.\n\n")
            pass  # Silently handle interruption
        except Exception as e:
            pass  # Silently handle loading errors
            self.chat_history.append(f"Error al cargar PDFs existentes: {str(e)}\n\n")

    def clear_database(self):
        reply = QMessageBox.question(self, 'Confirmar', '¿Estás seguro de que quieres limpiar toda la base de datos? Esta acción no se puede deshacer.',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                import sqlite3
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.execute('DELETE FROM pdfs')
                    conn.commit()
                # Clear Ollama knowledge base too
                self.ollama.knowledge_base = []
                # Clear RAG system
                self.rag_system.clear_knowledge_base()
                self.load_indexed_pdfs()
                self.chat_history.clear()
                self.chat_history.append("Base de datos y conocimiento indexado limpiados.\n\n")
                QMessageBox.information(self, 'Éxito', 'Base de datos limpiada correctamente.')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error al limpiar la base de datos: {str(e)}')

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
