import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# 1. Configuración de página con layout ancho
st.set_page_config(
    page_title="AuditIntel AI - Análisis Contable",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para emular la interfaz de la imagen
st.markdown("""
    <style>
    .banner-container {
        background: linear-gradient(90deg, #1E1035 0%, #321B63 50%, #22336E 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
    }
    .banner-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .banner-subtitle {
        font-size: 16px;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Barra Lateral (Sidebar)
with st.sidebar:
    # Cargar imagen de perfil/bot si existe
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, use_column_width=True)
    except Exception:
        pass

    st.markdown("### 🎯 AuditIntel AI")
    st.caption("Asistente RAG especializado en auditoría financiera, balances, estados de resultados y cumplimiento tributario.")

    st.divider()

    st.markdown("### 🔑 Autenticación")
    ke = st.text_input('Clave de API de OpenAI', type="password", placeholder="sk-...")

    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key cargada correctamente")
    else:
        st.warning("Ingresa tu API Key para habilitar la plataforma.")

    st.divider()
    st.caption(f"Versión de Python: {platform.python_version()}")

# 3. Encabezado principal tipo Banner
st.markdown("""
    <div class="banner-container">
        <div class="banner-title">Análisis de Auditoría & Estados Financieros 📊</div>
        <div class="banner-subtitle">Carga dictámenes, balances generales o informes de auditoría en PDF para extraer hallazgos clave en segundos.</div>
    </div>
""", unsafe_allow_html=True)

# 4. Sección superior dividida en dos columnas
col_upload, col_examples = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📁 Cargar Documento Financiero")
    st.caption("Selecciona un informe contable, auditoría o balance (PDF)")
    pdf = st.file_uploader("Subir PDF", type="pdf", label_visibility="collapsed")

with col_examples:
    st.markdown("### 💡 Ejemplos de Consultas")
    st.markdown("""
    * *"¿Cuáles son los hallazgos o salvedades principales expresados en el dictamen?"*
    * *"Resume los ingresos netos, costos y la utilidad operacional del periodo."*
    * *"¿Se mencionan pasivos contingentes o riesgos fiscales significativos?"*
    * *"Identifica las principales variaciones en el activo corriente respecto al periodo anterior."*
    """)

st.divider()

# 5. Flujo de procesamiento RAG
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        # Chunking
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=600,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Base de Conocimiento (FAISS)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.markdown("### 💬 Área de Consulta Financiera")
        user_question = st.text_input("Haz una pregunta sobre el documento auditado:", placeholder="Ej. ¿Cuál es el margen de utilidad operativa?")

        if user_question:
            with st.spinner("Analizando fuentes y generando respuesta contable..."):
                docs = knowledge_base.similarity_search(user_question)
                
                # Modelo ChatOpenAI actualizado
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                chain = load_qa_chain(llm, chain_type="stuff")
                
                response = chain.run(input_documents=docs, question=user_question)

                st.markdown("#### 📝 Hallazgos Extraídos:")
                st.info(response)

    except Exception as e:
        st.error(f"Error procesando el informe: {str(e)}")

elif pdf is None and ke:
    st.info("👋 Para comenzar, carga un archivo PDF de auditoría o finanzas desde el panel superior.")
elif pdf is not None and not ke:
    st.warning("⚠️ Por favor ingresa tu API Key en la barra lateral para procesar el documento.")
else:
    st.info("👋 Por favor ingresa tu API Key e ingresa un archivo PDF para empezar.")
