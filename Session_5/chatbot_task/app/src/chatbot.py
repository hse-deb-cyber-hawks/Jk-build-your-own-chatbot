import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.load.serializable import Serializable
from langchain_chroma import Chroma
from chromadb.api import ClientAPI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS 
import requests
import re
from uuid import uuid4
from typing import List, Tuple
import logging
import os
import datetime
from fpdf import FPDF 

OLLAMA_HOST_NAME = os.environ.get("OLLAMA_HOST_NAME", "localhost")
CHROMA_HOST_NAME = os.environ.get("CHROMA_HOST_NAME", "localhost")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3.2:1b")
PDF_DOC_PATH = os.environ.get("PDF_DOC_PATH", "src/AI_Book.pdf")

print("--- CHATBOT BACKEND GESTARTET ---")

class CustomChatBot:
    def __init__(self, index_data: bool, pull_embedding_model: bool) -> None:
        if pull_embedding_model:
            self._pull_embedding_model()

        self.embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=f"http://{OLLAMA_HOST_NAME}:11434")
        self.client = self._initialize_chroma_client()
        self.vector_db = self._initialize_vector_db()

        if index_data:
            self._index_data_to_vector_db()

        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatOllama(model=MODEL_NAME, base_url=f"http://{OLLAMA_HOST_NAME}:11434", temperature=0.2)
        
        # Standard: Fachexperte
        self.current_persona = "expert"
        self.qa_rag_chain = self._initialize_chain(persona="expert")

    def _pull_embedding_model(self):
        try:
            requests.post(f"http://{OLLAMA_HOST_NAME}:11434/api/pull", json={"name": EMBEDDING_MODEL, "stream": False})
        except:
            pass

    def _initialize_chroma_client(self) -> ClientAPI:
        return chromadb.HttpClient(host=CHROMA_HOST_NAME, port=8000, ssl=False, headers=None, settings=Settings(allow_reset=True, anonymized_telemetry=False), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE)

    def _initialize_vector_db(self) -> Chroma:
        return Chroma(client=self.client, collection_name="ai_model_book", embedding_function=self.embedding_function)

    def _index_data_to_vector_db(self):
        if os.path.exists(PDF_DOC_PATH):
            print(f"Lade Standard-Buch: {PDF_DOC_PATH}")
            self.ingest_new_file(PDF_DOC_PATH, "pdf")
        else:
            print("Standard-Buch nicht gefunden.")

    def ingest_new_file(self, file_path: str, file_type: str) -> Tuple[bool, str]:
        print(f"DEBUG: Versuche Datei zu laden: {file_path}")
        loader = None
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path)
            
        if not loader: return False, "Typ nicht unterstützt."

        try:
            raw_pages = loader.load()
            if not raw_pages: return False, "Datei leer."
            
            total_text = "".join([p.page_content for p in raw_pages])
            if len(total_text) < 50: return False, "Zu wenig Text (Scan?)."

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(raw_pages)
            docs = [Document(page_content=re.sub(r'[\ud800-\udfff]', '', d.page_content), metadata=d.metadata) for d in docs]
            
            if docs:
                ids = [str(uuid4()) for _ in range(len(docs))]
                self.vector_db.add_documents(documents=docs, ids=ids)
                print(f"DEBUG: {len(docs)} Chunks gespeichert.")
                return True, f"Erfolg! {len(docs)} Abschnitte gelernt."
            return False, "Fehler beim Splitten."
        except Exception as e:
            return False, f"Fehler: {str(e)}"

    # --- HIER IST DIE NEUE STIL-LOGIK ---
    def _initialize_chain(self, persona="expert") -> RunnableSerializable[Serializable, str]:
        
        # 1. FACHIDIOT (Anfänger / Laie)
        if persona == "beginner":
            sys_msg = """Du bist ein geduldiger Erklärer für komplette Anfänger (Laien).
            REGELN:
            - Antworte KURZ und KNAPP.
            - Nutze KEINE komplizierten Fachbegriffe. Wenn nötig, umschreibe sie einfach.
            - Nutze einfache Vergleiche aus dem Alltag.
            - Dein Ziel: Jeder, der noch nie davon gehört hat, muss es verstehen.
            """
            
        # 2. MITTEL (Fortgeschrittener)
        elif persona == "intermediate":
            sys_msg = """Du bist ein hilfreicher Tutor.
            REGELN:
            - Antworte ausführlicher als für Anfänger, aber verliere dich nicht in Details.
            - Nutze wichtige Fachbegriffe, aber erkläre sie kurz, falls sie komplex sind.
            - Deine Sprache soll präzise, aber gut lesbar und verständlich sein.
            - Biete eine gute Balance aus Fakten und Verständlichkeit.
            """
            
        # 3. FACHEXPERTE (Wissenschaftlich)
        else: # expert
            sys_msg = """Du bist ein hochqualifizierter Fachexperte und Wissenschaftler.
            REGELN:
            - Antworte ausführlich, detailliert und auf akademischem Niveau.
            - Nutze korrekte Fachterminologie (Termini Technici) ohne sie unnötig zu vereinfachen.
            - Bleibe strikt sachlich, objektiv und faktisch.
            - Deine Zielgruppe sind andere Experten oder Studenten.
            """

        system_instruction = f"""{sys_msg}
        
        GENERELLE ANWEISUNG:
        1. Nutze primär den KONTEXT (Dateien/Buch).
        2. Wenn Kontext leer -> Nutze dein Wissen, aber bleibe im gewählten Stil!
        3. Antworte direkt auf die Frage.
        """

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("system", "KONTEXT:\n{context}"),
            ("human", "VERLAUF:\n{history}\n\nFRAGE:\n{question}")
        ])
        
        return rag_prompt | self.llm | StrOutputParser()

    def set_persona(self, persona: str):
        self.current_persona = persona
        self.qa_rag_chain = self._initialize_chain(persona)

    async def astream(self, question: str, history: List[dict] = [], use_web: bool = False):
        context_text = ""
        if use_web:
            try:
                results = DDGS().text(question, max_results=3)
                if results: context_text = "\n".join([f"{r['title']}: {r['body']}" for r in results])
            except: pass
        else:
            try:
                docs = self.retriever.invoke(question)
                if docs: context_text = "\n\n".join(d.page_content for d in docs)
            except: pass

        history_text = "\n".join([f"User: {m.get('content', '')}" if m.get('role') == 'user' else f"Bot: {m.get('content', '')}" for m in history[-2:]])
        
        input_data = {"question": question, "context": context_text, "history": history_text}
        
        async for event in self.qa_rag_chain.astream_events(input_data, version="v2"):
            if event["event"] == "on_parser_stream": yield event["data"]["chunk"]

    def save_chat_to_pdf(self, chat_history: List[dict], filename: str = "Lernsession.pdf") -> str:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Protokoll", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        for msg in chat_history:
            role = "DU" if msg.get("role") == "user" else "BOT"
            content = str(msg.get("content", "")).encode('latin-1', 'replace').decode('latin-1')
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, f"{role}:", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, content)
            pdf.ln(5)
        path = f"/app/{filename}" if os.path.exists("/app") else filename
        try:
            pdf.output(path)
            return path
        except: return None

    def generate_quiz(self, text_input: str) -> str:
        template = """
        Erstelle ein Quiz mit 3 Fragen (Multiple Choice) basierend auf diesem KONTEXT:
        "{text_input}"
        
        Wenn der Text oben kein Wissen enthält, erstelle ein Quiz über "Künstliche Intelligenz".
        
        Format:
        1. Frage
        A) ...
        B) ...
        C) ...
        Lösung: ...
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text_input": text_input})