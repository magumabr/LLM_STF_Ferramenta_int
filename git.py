import os
import pandas as pd
from google import genai 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import PyPDF2
import streamlit as st
import uuid 

warnings.filterwarnings('ignore')

# ==========================================
# 0. CONFIGURAÇÃO E MEMÓRIA (SESSION STATE)
# ==========================================
st.set_page_config(page_title="Copiloto STI - Helpdesk", page_icon="⚖️", layout="wide")

# Inicialização do gerenciamento de estado (memória) para múltiplos chats simultâneos
if 'tela_atual' not in st.session_state:
    st.session_state.tela_atual = 'formulario'
if 'todas_conversas' not in st.session_state:
    st.session_state.todas_conversas = {} # Dicionário para armazenar o histórico de diferentes chamados
if 'chat_atual_id' not in st.session_state:
    st.session_state.chat_atual_id = None 
if 'mensagem_pendente' not in st.session_state:
    st.session_state.mensagem_pendente = None 

def aplicar_estilo_ui():
    """Aplica customizações CSS para deixar a interface do Streamlit mais limpa e profissional."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    div.stButton > button:first-child {
        background-color: #1E3A8A; color: white; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; border: none;
    }
    div.stButton > button:first-child:hover { background-color: #2563EB; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }
    .titulo-stf { font-weight: 600; color: #F8FAFC; border-bottom: 3px solid #EAB308; padding-bottom: 10px; margin-bottom: 20px; }
    
    /* Estilo para os botões da barra lateral (Histórico) */
    [data-testid="stSidebar"] div.stButton > button:first-child {
        background-color: transparent; color: #E2E8F0; text-align: left; border: 1px solid #334155; justify-content: flex-start;
    }
    [data-testid="stSidebar"] div.stButton > button:first-child:hover { background-color: #1E293B; border-color: #3B82F6; color: white; }
    </style>
    """, unsafe_allow_html=True)

aplicar_estilo_ui()

# ==========================================
# 1. IA E BASES DE CONHECIMENTO (RAG)
# ==========================================
# SEGURANÇA: Buscando a chave da API através de variáveis de ambiente em vez de hardcode
CHAVE_API_GEMINI = os.getenv("GEMINI_API_KEY", "COLOQUE_SUA_CHAVE_AQUI_SE_FOR_TESTAR_LOCAL") 
cliente_ia = genai.Client(api_key=CHAVE_API_GEMINI)

@st.cache_resource(show_spinner="Sincronizando Base de Conhecimento...")
def carregar_bases():
    """
    Carrega o CSV de histórico de chamados (Jira/ITSM) e os manuais (PDF/TXT) locais.
    Utiliza TF-IDF para vetorizar os textos e permitir buscas por similaridade (Cosine Similarity).
    """
    # Define caminhos relativos (facilita o deploy e uso por terceiros no GitHub)
    caminho_csv = './dados/historico_chamados_amostra.csv'
    pasta_kb = './dados/manuais'
    
    # --- Cérebro 1: Base de Histórico (CSV) ---
    try:
        df_historico = pd.read_csv(caminho_csv)
        if 'Campo personalizado (Link da base de conhecimento)' in df_historico.columns:
            df_historico = df_historico.rename(columns={'Campo personalizado (Link da base de conhecimento)': 'Link_POP'})
        else:
            df_historico['Link_POP'] = "Link não disponível."
        
        df_historico = df_historico.dropna(subset=['Descrição', 'Resolução'])
        vectorizer_hist = TfidfVectorizer(stop_words=None)
        matriz_hist = vectorizer_hist.fit_transform(df_historico['Descrição'].astype(str))
    except Exception as e:
        df_historico, vectorizer_hist, matriz_hist = None, None, None
        print(f"Aviso: Base de histórico não carregada. {e}")

    # --- Cérebro 2: Documentos e Manuais Oficiais ---
    documentos_kb = []
    if os.path.exists(pasta_kb):
        for arquivo in os.listdir(pasta_kb):
            caminho = os.path.join(pasta_kb, arquivo)
            texto_extraido = ""
            try:
                if arquivo.endswith('.txt'):
                    with open(caminho, 'r', encoding='utf-8') as f: texto_extraido = f.read()
                elif arquivo.endswith('.xlsx'):
                    df_excel = pd.read_excel(caminho)
                    texto_extraido = df_excel.to_string() 
                elif arquivo.endswith('.pdf'):
                    with open(caminho, 'rb') as f:
                        leitor_pdf = PyPDF2.PdfReader(f)
                        for pagina in leitor_pdf.pages: texto_extraido += pagina.extract_text() + "\n"
                if texto_extraido: documentos_kb.append({'nome': arquivo, 'texto': texto_extraido})
            except Exception: pass 

    vectorizer_kb, matriz_kb = None, None
    if documentos_kb:
        vectorizer_kb = TfidfVectorizer(stop_words=None)
        textos_kb = [doc['texto'] for doc in documentos_kb]
        matriz_kb = vectorizer_kb.fit_transform(textos_kb)

    return df_historico, vectorizer_hist, matriz_hist, documentos_kb, vectorizer_kb, matriz_kb

df_historico, vectorizer_hist, matriz_hist, documentos_kb, vectorizer_kb, matriz_kb = carregar_bases()

def buscar_contexto(nova_descricao, top_k=2):
    """Realiza a busca semântica nas bases de dados para embasar a IA."""
    contexto_final = ""
    if vectorizer_hist is not None:
        vetor_novo_hist = vectorizer_hist.transform([nova_descricao])
        sim_hist = cosine_similarity(vetor_novo_hist, matriz_hist).flatten()
        indices_hist = sim_hist.argsort()[-top_k:][::-1]
        contexto_final += "=== SOLUÇÕES DE CHAMADOS ANTIGOS ===\n"
        for idx in indices_hist:
            if sim_hist[idx] > 0.05: 
                contexto_final += f"Problema: {df_historico['Resumo'].iloc[idx]}\nResolução: {df_historico['Resolução'].iloc[idx]}\nLink: {df_historico['Link_POP'].iloc[idx]}\n\n"
                
    if vectorizer_kb is not None and documentos_kb:
        vetor_novo_kb = vectorizer_kb.transform([nova_descricao])
        sim_kb = cosine_similarity(vetor_novo_kb, matriz_kb).flatten()
        indices_kb = sim_kb.argsort()[-top_k:][::-1]
        contexto_final += "=== MANUAIS E POPS ===\n"
        for idx in indices_kb:
            if sim_kb[idx] > 0.01: 
                contexto_final += f"Arquivo: {documentos_kb[idx]['nome']}\nTrecho: {documentos_kb[idx]['texto'][:1500]}...\n\n" 
    return contexto_final

def gerar_resposta_inicial(titulo, descricao):
    """Gera o primeiro relatório de triagem estruturado."""
    contexto = buscar_contexto(descricao)
    prompt = f"""Você é o "Copiloto de Triagem de Helpdesk".
    DADOS DO CHAMADO: Título: {titulo} | Descrição: {descricao}
    CONTEXTO (Manuais e Histórico): {contexto}
    
    Escreva o relatório no formato:
    ### 📋 DIAGNÓSTICO RÁPIDO
    ### 🛠️ AÇÃO RECOMENDADA
    ### 🔗 LINKS E MANUAIS
    ### 🚀 ESCALONAMENTO"""
    return cliente_ia.models.generate_content(model='gemini-2.5-flash', contents=prompt).text

def gerar_resposta_chat(nova_mensagem, modo, historico):
    """Mantém a continuidade da conversa considerando o modo selecionado pelo analista."""
    contexto_conversa = "\n".join([f"{msg['role']}: {msg['content']}" for msg in historico])
    prompt = f"""Você é o Copiloto de Helpdesk.
    HISTÓRICO: {contexto_conversa}
    O analista enviou uma nova mensagem com o contexto: '{modo}'.
    Mensagem: {nova_mensagem}
    Regras:
    - Se "Informação Nova", o usuário final trouxe novos dados. Ajuste o diagnóstico.
    - Se "Dúvida para a IA", responda diretamente à dúvida técnica do analista."""
    return cliente_ia.models.generate_content(model='gemini-2.5-flash', contents=prompt).text


# ==========================================
# 2. BARRA LATERAL (GESTÃO DE SESSÕES)
# ==========================================
with st.sidebar:
    st.markdown("### Copiloto Helpdesk")
    
    if st.button("📝 Novo Chamado", type="primary", use_container_width=True):
        st.session_state.chat_atual_id = None
        st.session_state.tela_atual = 'formulario'
        st.rerun()
        
    st.write("---")
    st.markdown("**Chamados Recentes**")
    
    if not st.session_state.todas_conversas:
        st.caption("Nenhum chamado em análise.")
    else:
        for chat_id, chat_data in reversed(list(st.session_state.todas_conversas.items())):
            titulo_curto = chat_data['titulo'][:22] + "..." if len(chat_data['titulo']) > 22 else chat_data['titulo']
            icone = "📌" if chat_id == st.session_state.chat_atual_id else "💬"
            
            if st.button(f"{icone} {titulo_curto}", key=f"btn_{chat_id}", use_container_width=True):
                st.session_state.chat_atual_id = chat_id
                st.session_state.tela_atual = 'chat'
                st.rerun()

# ==========================================
# 3. TELA 1: FORMULÁRIO DE ENTRADA
# ==========================================
if st.session_state.tela_atual == 'formulario':
    st.markdown('<h1 class="titulo-stf">⚖️ Triagem Inteligente</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            titulo_input = st.text_input("Resumo/Título do Chamado:")
            descricao_input = st.text_area("Descrição Inicial:", height=150)
            
            if st.button("🚀 Processar com IA", use_container_width=True):
                if titulo_input and descricao_input:
                    with st.spinner("Buscando contexto na base de dados..."):
                        resposta = gerar_resposta_inicial(titulo_input, descricao_input)
                        
                        novo_id = str(uuid.uuid4())
                        st.session_state.todas_conversas[novo_id] = {
                            'titulo': titulo_input,
                            'historico': [
                                {"role": "user", "content": f"**Chamado Aberto:** {titulo_input}\n{descricao_input}"},
                                {"role": "ai", "content": resposta}
                            ]
                        }
                        st.session_state.chat_atual_id = novo_id
                        st.session_state.tela_atual = 'chat'
                        st.rerun() 
                else:
                    st.warning("Preencha os dados para iniciar.")

# ==========================================
# 4. TELA 2: MODO CHAT INTERATIVO
# ==========================================
elif st.session_state.tela_atual == 'chat':
    chat_id = st.session_state.chat_atual_id
    chat_atual = st.session_state.todas_conversas[chat_id]
    
    @st.dialog("Classifique o Contexto da Mensagem")
    def modal_confirmar_mensagem():
        msg_atual = st.session_state.mensagem_pendente
        st.markdown(f"**Mensagem:**\n> {msg_atual}")
        
        modo_escolhido = st.radio("Como a IA deve interpretar isso?", ["Informação Nova no Chamado", "Dúvida para a IA"], index=None)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Enviar", use_container_width=True):
                if modo_escolhido:
                    st.session_state.todas_conversas[chat_id]['historico'].append({"role": "user", "content": f"**[{modo_escolhido}]**\n{msg_atual}"})
                    with st.spinner("Analisando..."):
                        resposta_ia = gerar_resposta_chat(msg_atual, modo_escolhido, st.session_state.todas_conversas[chat_id]['historico'])
                    st.session_state.todas_conversas[chat_id]['historico'].append({"role": "ai", "content": resposta_ia})
                    st.session_state.mensagem_pendente = None
                    st.rerun()
                else:
                    st.error("Assinale uma opção.")
        with col2:
            if st.button("❌ Cancelar", use_container_width=True):
                st.session_state.mensagem_pendente = None
                st.rerun()

    st.markdown(f'<h1 class="titulo-stf">{chat_atual["titulo"]}</h1>', unsafe_allow_html=True)
    
    for msg in chat_atual['historico']:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"): st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"): st.markdown(msg["content"])

    if st.session_state.mensagem_pendente is not None:
        modal_confirmar_mensagem()

    nova_mensagem = st.chat_input("Insira uma nova interação ou dúvida...")
    if nova_mensagem:
        st.session_state.mensagem_pendente = nova_mensagem
        st.rerun()