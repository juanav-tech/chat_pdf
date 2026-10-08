import os
import platform
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st

# Imports actualizados para compatibilidad total con LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

# 1. Configuración de la página
st.set_page_config(
    page_title="Smart Audit RAG - Auditoría Financiera",
    page_icon="🔍",
    layout="wide"
)

# Estilo CSS para interfaz oscura con contraste
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
        margin-top: 15px;
    }
    .badge-tag {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Encabezado
col_title, col_logo = st.columns([3, 1])

with col_title:
    st.markdown('<span class="badge-tag">Auditoría & Análisis Contable</span>', unsafe_allow_html=True)
    st.title("🛡️ Smart Audit RAG")
    st.write("Análisis automatizado de informes financieros, dictámenes de auditoría, balances generales y cumplimiento fiscal.")

with col_logo:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=150)
    except Exception:
        pass

st.markdown("---")

# 3. Sidebar (Autenticación)
with st.sidebar:
    st.header("⚙️ Autenticación")
    ke = st.text_input('Clave API de OpenAI', type="password", placeholder="sk-...")
    
    st.markdown("---")
    st.subheader("📌 Instrucciones")
    st.markdown("""
    1. Ingresa tu API Key.
    2. Carga un reporte contable o financiero en PDF.
    3. Formula tus preguntas sobre salvedades, impuestos o ratios.
    """)
    st.caption(f"Python v{platform.python_version()}")

if ke:
    os.environ['OPENAI_API_KEY'] = ke

# 4. Disposición en dos columnas principales
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📄 1. Cargar Documento Financiero")
    pdf = st.file_uploader("Sube el PDF de auditoría o balance aquí", type="pdf")
    
    st.markdown("---")
    st.subheader("💡 Ejemplos de Consultas")
    st.info("""
    • ¿Cuáles son las salvedades o riesgos señalados por el auditor?
    • Resume los ingresos netos, la utilidad bruta y el EBITDA.
    • ¿Se detallan pasivos contingentes o litigios tributarios?
    • Identifica las principales variaciones en el activo corriente.
    """)

with col_right:
    st.subheader("🔍 2. Panel de Análisis")
    
    if not ke:
        st.warning("🔑 Ingresa tu API Key de OpenAI en la barra lateral para desbloquear el análisis.")
    elif pdf is None:
        st.info("👈 Sube un archivo PDF contable desde el panel izquierdo para comenzar.")
    else:
        try:
            # Lectura del texto del PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            
            st.success(f" Documento procesado correctamente ({len(text)} caracteres).")

            # Fragmentación del texto
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Base vectorial con FAISS y Embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Campo de consulta del usuario
            user_question = st.text_area(
                "Consulta sobre el informe contable:",
                placeholder="Ejemplo: Resume las principales notas a los estados financieros...",
                height=100
            )

            if user_question:
                with st.spinner("Analizando hallazgos financieros..."):
                    docs = knowledge_base.similarity_search(user_question)
                    
                    # LLM actualizado llamando desde langchain_openai
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    response = chain.run(input_documents=docs, question=user_question)

                    st.markdown("### 📊 Hallazgos de la Auditoría:")
                    st.markdown(f'<div class="metric-card">{response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
