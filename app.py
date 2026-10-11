import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Importaciones modernas de LangChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS (UX/UI FINANCIERO)
# ==========================================
st.set_page_config(
    page_title="AuditIntel Pro | Análisis Financiero & Auditoría",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 1.5rem 2rem; }
    .brand-header {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #4CC9F0;
    }
    .brand-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .brand-header p {
        color: #E0E1DD;
        font-size: 1.05rem;
        margin: 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1D4ED8 0%, #0284C7 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL (CONFIGURACIÓN & AUDITORÍA)
# ==========================================
with st.sidebar:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=280)
    except Exception:
        st.markdown("### 📊 AuditIntel Pro")

    st.markdown("## 🛡️ AuditIntel Pro")
    st.caption("Asistente RAG especializado en auditorías financieras, análisis de balances, estados de resultados y cumplimiento contable.")
    
    st.markdown("---")

    st.subheader("🔑 Autenticación")
    ke = st.text_input(
        "Clave de API de OpenAI", 
        type="password",
        placeholder="sk-...",
        help="Tu API Key es requerida para procesar embeddings y consultas financieras."
    )
    
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key vinculada correctamente")
    else:
        st.warning("Ingresa tu API Key para habilitar la plataforma.")

    st.markdown("---")
    st.caption(f"Entorno: Python v{platform.python_version()} | Engine: LangChain RAG")

# ==========================================
# CUERPO PRINCIPAL
# ==========================================
st.markdown("""
<div class="brand-header">
    <h1>Análisis Financiero & Auditoría Contable 📊</h1>
    <p>Carga estados de resultados, balances generales, flujos de caja o dictámenes de auditoría en PDF para extraer métricas y hallazgos clave en segundos.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Cargar Documento Financiero")
    pdf = st.file_uploader(
        "Selecciona un reporte contable, balance o dictamen (PDF)",
        type="pdf"
    )

with col2:
    st.subheader("💡 Ejemplos de Consultas Financieras")
    st.markdown("""
    * "¿Cuáles son las salvedades o notas clave expuestas en el dictamen del auditor?"
    * "Resume los ingresos netos, costos operativos y la utilidad neta del periodo."
    * "Identifica los principales riesgos de liquidez, solvencia o endeudamiento."
    * "¿Cuáles son las variaciones más significativas entre activos y pasivos?"
    """)

st.markdown("---")

# ==========================================
# LÓGICA DE PROCESAMIENTO RAG
# ==========================================
if pdf is not None and ke:
    try:
        with st.spinner("Procesando e indexando el documento financiero..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.success(f"¡Documento indexado con éxito! ({len(chunks)} bloques procesados)")
        
        st.subheader("🔍 Consultar al Asistente de Auditoría")
        
        with st.form(key="financial_query_form"):
            user_question = st.text_area(
                "Escribe tu pregunta de auditoría o análisis:",
                placeholder="Ejemplo: Realiza un resumen del estado de flujos de efectivo y el EBITDA...",
                height=100
            )
            
            submit_button = st.form_submit_button(label="🚀 Analizar y Generar Respuesta Financiera")

        if submit_button:
            if not user_question.strip():
                st.warning("Por favor escribe una consulta antes de procesar.")
            else:
                with st.spinner("Analizando la información contable y financiera del documento..."):
                    docs = knowledge_base.similarity_search(user_question, k=4)
                    context_text = "\n\n".join([doc.page_content for doc in docs])

                    llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")
                    
                    messages = [
                        ("system", "Eres un experto auditor financiero y analista contable. Responde la pregunta del usuario utilizando exclusivamente la información financiera y contable del contexto proporcionado."),
                        ("user", f"Contexto:\n{context_text}\n\nPregunta: {user_question}")
                    ]
                    
                    response = llm.invoke(messages)

                    st.markdown("---")
                    st.subheader("📋 Hallazgos & Respuesta de Auditoría")
                    st.info(response.content)

    except Exception as e:
        st.error("Ocurrió un problema al procesar el archivo o ejecutar la consulta financiera.")
        with st.expander("Ver detalle técnico del error"):
            st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Debes ingresar tu Clave de API de OpenAI en la barra lateral para continuar.")
else:
    st.info("👋 Para comenzar, carga un archivo PDF financiero desde el panel superior.")import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Importaciones modernas de LangChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS (UX/UI FINANCIERO)
# ==========================================
st.set_page_config(
    page_title="AuditIntel Pro | Análisis Financiero & Auditoría",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 1.5rem 2rem; }
    .brand-header {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #4CC9F0;
    }
    .brand-header h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .brand-header p {
        color: #E0E1DD;
        font-size: 1.05rem;
        margin: 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1D4ED8 0%, #0284C7 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.92; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BARRA LATERAL (CONFIGURACIÓN & AUDITORÍA)
# ==========================================
with st.sidebar:
    try:
        image = Image.open('Chat_pdf.png')
        st.image(image, width=280)
    except Exception:
        st.markdown("### 📊 AuditIntel Pro")

    st.markdown("## 🛡️ AuditIntel Pro")
    st.caption("Asistente RAG especializado en auditorías financieras, análisis de balances, estados de resultados y cumplimiento contable.")
    
    st.markdown("---")

    st.subheader("🔑 Autenticación")
    ke = st.text_input(
        "Clave de API de OpenAI", 
        type="password",
        placeholder="sk-...",
        help="Tu API Key es requerida para procesar embeddings y consultas financieras."
    )
    
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key vinculada correctamente")
    else:
        st.warning("Ingresa tu API Key para habilitar la plataforma.")

    st.markdown("---")
    st.caption(f"Entorno: Python v{platform.python_version()} | Engine: LangChain RAG")

# ==========================================
# CUERPO PRINCIPAL
# ==========================================
st.markdown("""
<div class="brand-header">
    <h1>Análisis Financiero & Auditoría Contable 📊</h1>
    <p>Carga estados de resultados, balances generales, flujos de caja o dictámenes de auditoría en PDF para extraer métricas y hallazgos clave en segundos.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Cargar Documento Financiero")
    pdf = st.file_uploader(
        "Selecciona un reporte contable, balance o dictamen (PDF)",
        type="pdf"
    )

with col2:
    st.subheader("💡 Ejemplos de Consultas Financieras")
    st.markdown("""
    * "¿Cuáles son las salvedades o notas clave expuestas en el dictamen del auditor?"
    * "Resume los ingresos netos, costos operativos y la utilidad neta del periodo."
    * "Identifica los principales riesgos de liquidez, solvencia o endeudamiento."
    * "¿Cuáles son las variaciones más significativas entre activos y pasivos?"
    """)

st.markdown("---")

# ==========================================
# LÓGICA DE PROCESAMIENTO RAG
# ==========================================
if pdf is not None and ke:
    try:
        with st.spinner("Procesando e indexando el documento financiero..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.success(f"¡Documento indexado con éxito! ({len(chunks)} bloques procesados)")
        
        st.subheader("🔍 Consultar al Asistente de Auditoría")
        
        with st.form(key="financial_query_form"):
            user_question = st.text_area(
                "Escribe tu pregunta de auditoría o análisis:",
                placeholder="Ejemplo: Realiza un resumen del estado de flujos de efectivo y el EBITDA...",
                height=100
            )
            
            submit_button = st.form_submit_button(label="🚀 Analizar y Generar Respuesta Financiera")

        if submit_button:
            if not user_question.strip():
                st.warning("Por favor escribe una consulta antes de procesar.")
            else:
                with st.spinner("Analizando la información contable y financiera del documento..."):
                    docs = knowledge_base.similarity_search(user_question, k=4)
                    context_text = "\n\n".join([doc.page_content for doc in docs])

                    llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")
                    
                    messages = [
                        ("system", "Eres un experto auditor financiero y analista contable. Responde la pregunta del usuario utilizando exclusivamente la información financiera y contable del contexto proporcionado."),
                        ("user", f"Contexto:\n{context_text}\n\nPregunta: {user_question}")
                    ]
                    
                    response = llm.invoke(messages)

                    st.markdown("---")
                    st.subheader("📋 Hallazgos & Respuesta de Auditoría")
                    st.info(response.content)

    except Exception as e:
        st.error("Ocurrió un problema al procesar el archivo o ejecutar la consulta financiera.")
        with st.expander("Ver detalle técnico del error"):
            st.code(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Debes ingresar tu Clave de API de OpenAI en la barra lateral para continuar.")
else:
    st.info("👋 Para comenzar, carga un archivo PDF financiero desde el panel superior.")
