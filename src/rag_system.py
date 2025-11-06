from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import os
import json

class RAGSystem:
    def __init__(self, ollama_base_url='http://10.65.117.238:11434', model='granite4:tiny-h'):
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.qa_chain = None
        self.documents = [] 

        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=model
        )

    def add_pdf_document(self, filename, content, metadata=None):
        """Add a PDF document to the RAG system"""
        # Create document with metadata
        doc_metadata = {
            'filename': filename,
            'source': 'pdf_indexer'
        }
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                doc_metadata.update(parsed_metadata)
            except:
                pass

        # Split text into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(content)
        doc_chunks = [
            Document(page_content=chunk, metadata={**doc_metadata, 'chunk_id': i})
            for i, chunk in enumerate(chunks)
        ]

        self.documents.extend(doc_chunks)

        # Rebuild vectorstore with new documents
        self._rebuild_vectorstore()

        return f"Documento '{filename}' agregado al sistema RAG. Chunks: {len(doc_chunks)}"

    def _rebuild_vectorstore(self):
        """Rebuild the FAISS vectorstore with current documents"""
        if self.documents:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

            # Create QA chain using LCEL (LangChain Expression Language)
            template = """Usa la siguiente información para responder la pregunta del usuario.
            Si no sabes la respuesta, di que no lo sabes, no inventes información.

            Información relevante:
            {context}

            Pregunta: {question}

            Respuesta detallada:"""

            prompt = PromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.qa_chain = (
                {"context": self.vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
                 "question": RunnablePassthrough()}
                | prompt
                | self.llm
            )

    def query(self, question):
        """Query the RAG system"""
        if not self.qa_chain:
            return "No hay documentos indexados en el sistema RAG."

        try:
            # Get response using LCEL chain
            response = self.qa_chain.invoke(question)

            # Get relevant documents for source information
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            source_docs = retriever.invoke(question)

            # Add source information
            if source_docs:
                sources = []
                for doc in source_docs[:3]:  # Show top 3 sources
                    filename = doc.metadata.get('filename', 'Desconocido')
                    sources.append(filename)

                response += f"\n\nFuentes consultadas: {', '.join(set(sources))}"

            return response

        except Exception as e:
            return f"Error en consulta RAG: {str(e)}"

    def get_document_count(self):
        """Get total number of indexed documents"""
        return len(self.documents)

    def clear_knowledge_base(self):
        """Clear all indexed documents"""
        self.documents = []
        self.vectorstore = None
        self.qa_chain = None

    def save_vectorstore(self, path="vectorstore"):
        """Save vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path="vectorstore"):
        """Load vectorstore from disk"""
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            # Rebuild QA chain with loaded vectorstore
            self._rebuild_vectorstore()
            return True
        except:
            return False
