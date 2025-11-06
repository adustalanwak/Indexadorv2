import requests
import json

class OllamaClient:
    def __init__(self, base_url='http://10.65.117.238:11434', model='granite4:tiny-h'):
        self.base_url = base_url
        self.model = model
        self.knowledge_base = []  # Store indexed PDF content for learning
        self.available_models = self.get_available_models()  # Get available models on init

    def index_pdf_content(self, pdf_text, metadata=None):
        """Index PDF content for future queries - simulates learning"""
        entry = {
            'content': pdf_text,
            'metadata': metadata or {},
            'indexed': True
        }
        self.knowledge_base.append(entry)
        return f"PDF indexado en la base de conocimiento. Total de documentos: {len(self.knowledge_base)}"

    def query(self, prompt, context="", images=None):
        # Include indexed knowledge base in context
        full_context = context
        if self.knowledge_base:
            full_context += "\n\n--- CONOCIMIENTO INDEXADO ---\n"
            for i, entry in enumerate(self.knowledge_base, 1):
                full_context += f"Documento {i}:\n{entry['content'][:1000]}...\n"
                if entry['metadata']:
                    full_context += f"Metadatos: {json.dumps(entry['metadata'], ensure_ascii=False)}\n\n"

        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": f"Contexto: {full_context}\n\nPregunta: {prompt}\n\nResponde de manera detallada sobre la información.",
            "stream": False
        }

        # Add images if provided (for vision models)
        if images:
            data["images"] = images

        try:
            response = requests.post(url, json=data)  # Removed timeout for unlimited response time
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'No response from Ollama')
        except requests.RequestException as e:
            return f"Error querying Ollama: {str(e)}"

    def get_indexed_count(self):
        """Get the number of indexed documents"""
        return len(self.knowledge_base)

    def get_available_models(self):
        """Get list of available models from Ollama server"""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except requests.RequestException as e:
            pass  # Silently handle model fetching failure
            return ['granite4:tiny-h']  # fallback to default model

    def get_available_models_list(self):
        """Return the cached list of available models"""
        return self.available_models if self.available_models else ['granite4:tiny-h']
