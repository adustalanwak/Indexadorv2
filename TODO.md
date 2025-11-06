# TODO List for PDF Indexer and Chat App

## 1. Set up project structure
- [x] Create main directories: src/, db/, pdfs/ (for temporary storage if needed)
- [x] Create main files: main.py, gui.py, db.py, pdf_processor.py, ollama_client.py, requirements.txt

## 2. Install dependencies
- [x] Add PyQt6, pdfplumber, pytesseract, sqlite3, requests to requirements.txt
- [x] Run pip install -r requirements.txt
- [x] Update LangChain dependencies (langchain-huggingface, langchain-ollama)

## 3. Implement database module (db.py)
- [x] Create SQLite database schema for storing PDF metadata and extracted text
- [x] Functions to insert, query PDFs

## 4. Implement PDF processor (pdf_processor.py)
- [x] Function to extract text from PDFs (using pdfplumber)
- [x] Handle scanned PDFs with OCR (pytesseract)
- [x] Extract specific fields for Mexican tax documents (declarations, invoices, etc.)

## 5. Implement Ollama client (ollama_client.py)
- [x] Function to send queries to Ollama server at 10.65.117.238:11434
- [x] Use granite4:tiny-h model
- [x] Parse responses for summaries

## 6. Implement RAG system (rag_system.py)
- [x] Implement LangChain-based RAG system with FAISS vectorstore
- [x] Integrate with Ollama for question answering
- [x] Add document chunking and semantic search

## 7. Implement GUI (gui.py)
- [x] Use PyQt6 for a nice-looking interface
- [x] Folder selection dialog
- [x] List indexed PDFs
- [x] Chat interface for queries with RAG
- [x] Display summaries with source information

## 7. Integrate all modules in main.py
- [x] Initialize database
- [x] Launch GUI
- [x] Handle indexing process

## 8. Test the application
- [x] Test LangChain dependencies installation
- [x] Test RAG system creation and document addition
- [x] Test database and PDF processor modules
- [x] Test Ollama client
- [x] Test GUI imports
- [x] Test main module import
- [x] Test database operations (insert, query)
- [x] Test PDF processor with sample data
- [x] Test Ollama client queries
- [x] Test GUI launch and basic functionality
- [ ] Index a sample folder with PDFs
- [ ] Query for summaries
- [ ] Ensure GUI works smoothly
- [x] Final integration test (run the complete application)

## 9. Mejoras implementadas
- [x] Agregadas librerías avanzadas: pdfminer.six, camelot-py, opencv-python
- [x] Mejor extracción de texto con pdfminer.six como primera opción
- [x] Extracción de tablas con camelot
- [x] Campos adicionales en facturas: emisor, receptor, subtotal, IVA, retenciones
- [x] Patrones de regex mejorados para facturas mexicanas
- [x] Preprocesamiento de imágenes para mejor OCR
- [x] Mostrar datos estructurados en la interfaz
- [x] Botones para limpiar selección y base de datos
- [x] Timeout removido para respuestas ilimitadas de Ollama
- [x] Sistema de aprendizaje: PDFs indexados en base de conocimiento de Ollama

## 10. Correcciones recientes
- [x] Corregir imports deprecated de LangChain (HuggingFaceEmbeddings, OllamaLLM)
- [x] Actualizar requirements.txt con langchain-huggingface y langchain-ollama
- [x] Corregir método retriever (invoke en lugar de get_relevant_documents)
- [x] Solucionar problemas de bloqueo de archivos con archivos temporales en PDF processor
- [x] Hardcodear IP del servidor Ollama (10.65.117.238:11434) en la interfaz

## 11. Seguridad implementada
- [x] Agregar herramienta Safety para escaneo de vulnerabilidades
- [x] Actualizar dependencias vulnerables (Pillow, requests, langchain-community, langchain-core)
- [x] Verificar que no hay vulnerabilidades conocidas en las dependencias

## 12. Mejoras de generalización
- [x] Hacer el extractor de datos más genérico para cualquier tipo de PDF
- [x] Mantener RFC como campo primordial para documentos mexicanos
- [x] Agregar patrones para contratos, licencias, estados de cuenta, etc.
- [x] Incluir información de contacto (email, teléfono, dirección)
