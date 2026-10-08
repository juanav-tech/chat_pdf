import os
import platform
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st

# Imports modernos y compatibles
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# Configuración de la página
st.set_page_config(
    page_title="Smart Audit RAG - Auditoría Financiera",
    page_icon="🔍",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #E6E6E6;
    }
    .metric-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363D;
        margin-bottom: 15px;
    }
    .badge-tag {
        background-color: #1F6FEB;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Encabezado principal
col_title, col_logo = st.columns([3, 1])

with col_title:
    st.markdown('<span class="badge-tag">Módulo de Auditoría Contable</span>', unsafe_allow_html=True)
    st.title("🛡️ Smart Audit RAG")
    st.write("Auditoría inteligente de informes financieros, balances generales, estados de resultados y dictámenes tributarios.")

with col_logo:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=160)
    except Exception:
        pass

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    ke = st.text_input('Clave API de OpenAI', type="password", placeholder="sk-...")
    
    st.markdown("---")
    st.subheader("📌 Guía de Uso")
    st.markdown("""
    1. Ingresa tu API Key.
    2. Carga un reporte o balance contable en PDF.
    3. Realiza preguntas sobre salvedades, impuestos o ratios.
    """)
    st.caption(f"Python v{platform.python_version()}")

if ke:
    os.environ['OPENAI_API_KEY'] = ke

# Estructura principal
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📄 1. Cargar Documento PDF")
    pdf = st.file_uploader("Arrastra tu informe financiero aquí", type="pdf")
    
    st.markdown("---")
    st.subheader("💡 Consultas Sugeridas para Auditoría")
    st.info("""
    • ¿Cuáles son las salvedades expuestas en el dictamen del auditor?
    • Resume el margen bruto, la utilidad operativa y el EBITDA.
    • ¿Se identifican pasivos contingentes o litigios pendientes?
    • ¿Qué riesgos fiscales o de liquidez señala el documento?
    """)

with col_right:
    st.subheader("🔍 2. Panel de Consulta")
    
    if not ke:
        st.warning("🔑 Ingresa tu API Key de OpenAI en el menú lateral para desbloquear el análisis.")
    elif pdf is None:
        st.info("👈 Sube un archivo PDF contable desde el panel izquierdo para comenzar.")
    else:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            
            st.success(f"Documento leído con éxito ({len(text)} caracteres).")

            # Separador usando la nueva librería langchain_text_splitters
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            user_question = st.text_area(
                "Pregunta sobre el documento auditado:",
                placeholder="Ejemplo: Realiza un resumen del estado de pérdidas y ganancias...",
                height=100
            )

            if user_question:
                with st.spinner("Analizando hallazgos financieros..."):
                    docs = knowledge_base.similarity_search(user_question)
                    
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    response = chain.run(input_documents=docs, question=user_question)

                    st.markdown("### 📊 Resultado de la Auditoría:")
                    st.markdown(f'<div class="metric-card">{response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error procesando el PDF: {str(e)}")
