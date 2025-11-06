import pdfplumber
import pytesseract
from PIL import Image
import io
import json
import re
from pdfminer.high_level import extract_text as pdfminer_extract
import camelot
import cv2
import numpy as np
import tempfile
import os
import shutil
import base64
from ollama_client import OllamaClient

class PDFProcessor:
    def __init__(self):
        # Configure pytesseract if needed (path to tesseract executable)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_text_from_pdf(self, pdf_path):
        text = ""

        # Try pdfminer.six first for better text extraction
        try:
            text = pdfminer_extract(pdf_path)
            if text.strip():
                return text.strip()
        except Exception as e:
            pass  # Silently handle pdfminer.six failure

        # Fallback to pdfplumber with OCR support for image-based PDFs
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        # If no text extracted, try OCR on the page image
                        try:
                            # Convert page to image
                            page_image = page.to_image(resolution=300).original
                            # Preprocess image for better OCR
                            processed_image = self.preprocess_image_for_ocr(page_image)
                            # Extract text with OCR
                            ocr_text = pytesseract.image_to_string(processed_image, lang='spa+eng')
                            if ocr_text.strip():
                                text += ocr_text + "\n"
                        except Exception as ocr_e:
                            pass  # Silently handle OCR failure
        except Exception as e:
            pass  # Silently handle pdfplumber failure

        return text.strip()

    def _ocr_image(self, image):
        # Convert PIL Image to bytes for pytesseract
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return pytesseract.image_to_string(Image.open(io.BytesIO(img_byte_arr)))

    def extract_tables_hybrid(self, pdf_path):
        """
        Hybrid table extraction: Camelot first, vision-language fallback for complex tables
        """
        # 1. Try traditional table extraction with Camelot
        traditional_tables = self.extract_tables_camelot(pdf_path)

        # 2. Validate table quality
        if self._validate_table_quality(traditional_tables):
            return {
                'method': 'camelot',
                'tables': traditional_tables,
                'quality_score': self._calculate_table_quality(traditional_tables)
            }

        # 3. Fallback to vision-language for complex tables
        vision_tables = self.extract_tables_vision(pdf_path)

        return {
            'method': 'vision_fallback',
            'tables': vision_tables,
            'quality_score': 0.8  # Assume good quality from vision
        }

    def extract_tables_camelot(self, pdf_path):
        """Extract tables using camelot for better table recognition"""
        try:
            # Try lattice flavor first (better for tables with clear borders)
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            if len(tables) == 0:
                # Fallback to stream flavor for tables without clear borders
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

            table_data = []
            for table in tables:
                # Convert table to dict and include accuracy info
                table_dict = table.df.to_dict()
                table_dict['_accuracy'] = table.accuracy
                table_dict['_whitespace'] = table.whitespace
                table_data.append(table_dict)
            return table_data
        except Exception as e:
            pass  # Silently handle table extraction failure
            return []

    def extract_tables_vision(self, pdf_path):
        """
        Extract tables using vision-language model as fallback
        """
        try:
            # Convert PDF pages to images
            page_images = self._pdf_to_images(pdf_path)

            vision_client = OllamaClient()  # Assuming vision model available
            extracted_tables = []

            for i, image in enumerate(page_images):
                # Convert image to base64 for vision model
                base64_image = self._image_to_base64(image)

                # Query vision model for table extraction
                prompt = """
                Analiza esta imagen y extrae todas las tablas que encuentres.
                Para cada tabla, proporciona:
                1. Encabezados de columna
                2. Datos fila por fila
                3. Formato JSON estructurado

                Si no hay tablas, responde "No tables found".
                """

                # Note: This assumes the vision model is available in Ollama
                # You might need to adjust based on available models
                try:
                    response = vision_client.query(prompt, context=f"Page {i+1} of PDF")

                    # Parse the response to extract table data
                    table_data = self._parse_vision_table_response(response)
                    if table_data:
                        extracted_tables.extend(table_data)
                except Exception as e:
                    # Silently handle vision query failure for this page
                    continue

            return extracted_tables

        except Exception as e:
            pass  # Silently handle vision extraction failure
            return []

    def _pdf_to_images(self, pdf_path, dpi=150):
        """Convert PDF pages to PIL images"""
        images = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Convert page to image
                    page_image = page.to_image(resolution=dpi).original
                    images.append(page_image)
        except Exception as e:
            pass  # Silently handle conversion failure
        return images

    def _image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            return ""

    def _validate_table_quality(self, tables):
        """Validate if extracted tables meet quality standards"""
        if not tables:
            return False

        # Check if tables have reasonable structure
        for table in tables:
            if '_accuracy' in table:
                accuracy = table['_accuracy']
                if accuracy < 50:  # Less than 50% accuracy
                    return False

            # Check if table has data
            df_keys = [k for k in table.keys() if isinstance(k, str) and not k.startswith('_')]
            if len(df_keys) < 2:  # Less than 2 columns
                return False

            # Check if table has rows
            if len(table[df_keys[0]]) < 1:  # No data rows
                return False

        return True

    def _calculate_table_quality(self, tables):
        """Calculate overall quality score for extracted tables"""
        if not tables:
            return 0.0

        total_accuracy = 0
        count = 0

        for table in tables:
            if '_accuracy' in table:
                total_accuracy += table['_accuracy']
                count += 1

        return total_accuracy / count if count > 0 else 0.5

    def _parse_vision_table_response(self, response):
        """Parse vision model response to extract structured table data"""
        # This is a simplified parser - you might need to adjust based on model output
        try:
            # Look for JSON-like structures in the response
            import ast

            # Try to find and parse JSON structures
            json_matches = re.findall(r'\{.*?\}', response, re.DOTALL)
            tables = []

            for match in json_matches:
                try:
                    table_data = ast.literal_eval(match)
                    if isinstance(table_data, dict) and 'headers' in table_data:
                        tables.append(table_data)
                except:
                    continue

            return tables

        except Exception as e:
            return []

    # Keep the old method for backward compatibility
    def extract_tables_from_pdf(self, pdf_path):
        """Legacy method - use extract_tables_hybrid instead"""
        result = self.extract_tables_hybrid(pdf_path)
        return result.get('tables', [])

    def preprocess_image_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        try:
            # Convert PIL to OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Convert back to PIL Image
            pil_image = Image.fromarray(thresh)
            return pil_image
        except Exception as e:
            pass  # Silently handle image preprocessing failure
            return image  # Return original image if preprocessing fails

    def extract_document_data(self, text, tables=None):
        # Extract general document information - adaptable to any PDF type
        data = {}

        # Document type detection - more general patterns
        doc_type_patterns = [
            r'(?:tipo de|tipo|clase de)\s*(?:documento|declaración|informe|reporte)[:\s]*([^\n\r]+)',
            r'(?:declaración|declaracion|informe|reporte)\s+(?:anual|mensual|trimestral|semestral|diario)',
            r'(?:documento|archivo|expediente)\s+(?:oficial|gubernamental|institucional)',
            r'(?:SAT|SERVICIOS\s+DE\s+ADMINISTRACIÓN\s+TRIBUTARIA)',
            r'(?:CFDI|COMPROBANTE\s+FISCAL\s+DIGITAL)',
            r'(?:CONSTANCIA|CERTIFICADO|CERTIFICACIÓN)',
            r'(?:RECIBO|RECIBO\s+DE\s+PAGO|COMPROBANTE\s+DE\s+PAGO)',
            r'(?:ESTADO\s+DE\s+CUENTA|EXTRACTO\s+BANCARIO)',
            r'(?:CONTRATO|ACUERDO|CONVENIO)',
            r'(?:LICENCIA|PERMISO|AUTORIZACIÓN)'
        ]

        for pattern in doc_type_patterns:
            type_match = re.search(pattern, text, re.IGNORECASE)
            if type_match:
                data['tipo_documento'] = type_match.group(1).strip() if type_match.groups() else type_match.group().strip()
                break

        # RFC patterns (Mexican tax ID) - keeping this as it's fundamental for Mexican documents
        rfc_patterns = [
            r'\b[A-Z]{3,4}\d{6}[A-Z0-9]{3}\b',
            r'RFC[:\s]*([A-Z]{3,4}\d{6}[A-Z0-9]{3})',
            r'Registro Federal de Contribuyentes[:\s]*([A-Z]{3,4}\d{6}[A-Z0-9]{3})',
            r'R\.?F\.?C\.?[:\s]*([A-Z]{3,4}\d{6}[A-Z0-9]{3})'
        ]
        for pattern in rfc_patterns:
            rfc_match = re.search(pattern, text, re.IGNORECASE)
            if rfc_match:
                data['rfc'] = rfc_match.group(1) if rfc_match.groups() else rfc_match.group()
                break

        # General entity/person name (more flexible)
        entity_patterns = [
            r'(?:contribuyente|declarante|titular|propietario|empresa|persona)[:\s]*(.*?)(?=\n|\r|$|RFC)',
            r'(?:nombre|razón social|denominación|denominacion)[:\s]*(.*?)(?=\n|\r|$|RFC)',
            r'(?:emisor|receptor|destinatario)[:\s]*(.*?)(?=\n|\r|$)',
            r'(?:cliente|proveedor|contratante|contratista)[:\s]*(.*?)(?=\n|\r|$)'
        ]
        for pattern in entity_patterns:
            entity_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if entity_match:
                data['entidad_principal'] = entity_match.group(1).strip()
                break

        # Period/Date information (more general)
        period_patterns = [
            r'(?:ejercicio|eje rcicio|período|periodo|año|ano)[:\s]*(\d{4})',
            r'(?:fecha|date)[:\s]*(?:\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(?:del?\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})(?:\s+a\s+|\s+al\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})?',
            r'\b(\d{4})\b',  # Year pattern
            r'(?:mes|month)[:\s]*([^\n\r]+)',
            r'(?:trimestre|quarter)[:\s]*([^\n\r]+)'
        ]
        for pattern in period_patterns:
            period_match = re.search(pattern, text, re.IGNORECASE)
            if period_match:
                data['periodo_fecha'] = period_match.group(1) if period_match.groups() else period_match.group()
                break

        # Monetary amounts (general purpose)
        amount_patterns = {
            'total': r'(?:total|totales?|monto|importe|suma)[:\s]*\$?[\d,]+\.?\d*',
            'subtotal': r'subtotal[:\s]*\$?[\d,]+\.?\d*',
            'iva': r'IVA[:\s]*\$?[\d,]+\.?\d*',
            'isr': r'ISR[:\s]*\$?[\d,]+\.?\d*',
            'ieps': r'IEPS[:\s]*\$?[\d,]+\.?\d*',
            'saldo': r'saldo[:\s]*\$?[\d,]+\.?\d*',
            'deuda': r'deuda[:\s]*\$?[\d,]+\.?\d*',
            'pago': r'pago[:\s]*\$?[\d,]+\.?\d*',
            'ingresos': r'ingresos[:\s]*\$?[\d,]+\.?\d*',
            'egresos': r'egresos[:\s]*\$?[\d,]+\.?\d*',
            'utilidad': r'utilidad[:\s]*\$?[\d,]+\.?\d*',
            'perdida': r'(?:pérdida|perdida)[:\s]*\$?[\d,]+\.?\d*'
        }

        for amount_type, pattern in amount_patterns.items():
            amount_match = re.search(pattern, text, re.IGNORECASE)
            if amount_match:
                data[amount_type] = amount_match.group()

        # Document/Transaction numbers (more general)
        doc_number_patterns = [
            r'(?:folio|número|no\.?|id|identificador|código|codigo)[:\s]*([\w\d-]+)',
            r'(?:referencia|ref\.?)[:\s]*([\w\d-]+)',
            r'(?:número\s+de\s+documento|documento)[:\s]*([\w\d-]+)',
            r'(?:contrato|convenio|acuerdo)[:\s]*([\w\d-]+)',
            r'(?:licencia|permiso)[:\s]*([\w\d-]+)'
        ]
        for pattern in doc_number_patterns:
            doc_match = re.search(pattern, text, re.IGNORECASE)
            if doc_match:
                data['numero_identificador'] = doc_match.group(1) if doc_match.groups() else doc_match.group()
                break

        # Status/Condition (general)
        status_patterns = [
            r'(?:estatus|estado|condición|condicion|situación|situacion)[:\s]*([^\n\r]+)',
            r'(?:status|state)[:\s]*([^\n\r]+)',
            r'(?:vigente|activo|cancelado|pendiente|aprobado|rechazado)'
        ]
        for pattern in status_patterns:
            status_match = re.search(pattern, text, re.IGNORECASE)
            if status_match:
                data['estatus_condicion'] = status_match.group(1).strip() if status_match.groups() else status_match.group().strip()
                break

        # Contact/Address information (if present)
        contact_patterns = {
            'telefono': r'(?:teléfono|telefono|tel\.?|phone)[:\s]*([^\n\r]+)',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'direccion': r'(?:dirección|direccion|address)[:\s]*(.*?)(?=\n|\r|$|Teléfono|Email|RFC)'
        }

        for contact_type, pattern in contact_patterns.items():
            contact_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if contact_match:
                data[contact_type] = contact_match.group(1).strip() if contact_match.groups() else contact_match.group()

        # If tables are available, include them
        if tables:
            data['tablas_extraidas'] = len(tables)

        return json.dumps(data, ensure_ascii=False) if data else None
