import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
from sentence_transformers import SentenceTransformer
import os

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 1. Load LLM + Embedding Model (cached)
# ============================================================
@st.cache_resource
def load_llm():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"   # 🔥 CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,            # 🔥 FIX: FP16 breaks CPU on Streamlit
        device_map=None,                # 🔥 FIX: avoid accelerate/GPU mapping
        low_cpu_mem_usage=True
    )
    return tokenizer, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer, model = load_llm()
embedder = load_embedder()

# ============================================================
# 2. Sidebar
# ============================================================
st.sidebar.title("⚙️ Settings")
thinking_mode = st.sidebar.checkbox("Enable Thinking Mode", value=False)

languages = {
    "English": "en",
    "Malay": "ms",
    "Indonesian": "id",
    "Thai": "th",
    "Vietnamese": "vi",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Burmese": "my",
}
selected_language = st.sidebar.selectbox("App Language", list(languages.keys()))

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a multilingual helpful assistant.",
    height=120
)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF/TXT)",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

# ============================================================
# 3. RAG: Vector Store
# ============================================================
if "doc_index" not in st.session_state:
    st.session_state.doc_index = None
if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = []

def process_uploaded_files(files):
    import PyPDF2

    all_texts = []
    for f in files:
        if f.type == "text/plain":
            content = f.read().decode("utf-8")
        elif f.type == "application/pdf":
            reader = PyPDF2.PdfReader(f)
            content = "\n".join([p.extract_text() for p in reader.pages])
        else:
            continue
        chunks = chunk_text(content)
        all_texts.extend(chunks)

    if not all_texts:
        return None, None

    embeddings = embedder.encode(all_texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, all_texts

def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

if uploaded_files:
    st.sidebar.write("Processing documents…")
    index, texts = process_uploaded_files(uploaded_files)
    st.session_state.doc_index = index
    st.session_state.doc_texts = texts
    st.sidebar.success("RAG data ready!")

# ============================================================
# 4. Chat History
# ============================================================
st.title("🦁 SEA-LION Enhanced Chat App (CPU Compatible)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# ============================================================
# 5. RAG Retrieval
# ============================================================
def retrieve(query, k=3):
    if st.session_state.doc_index is None:
        return ""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    scores, idxs = st.session_state.doc_index.search(q_emb, k)
    return "\n\n".join(st.session_state.doc_texts[i] for i in idxs[0])

# ============================================================
# 6. Chat Interface
# ============================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

user_input = st.chat_input("Type your message…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    rag_context = retrieve(user_input)
    rag_block = f"\n\n[Context]\n{rag_context}\n\n" if rag_context else ""

    messages_for_model = [
        st.session_state.messages[0],
        *st.session_state.messages[1:]
    ]
    if rag_block:
        messages_for_model[-1]["content"] += rag_block

    prompt_text = tokenizer.apply_chat_template(
        messages_for_model,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt_text], return_tensors="pt")

    with st.spinner("Generating response…"):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
        )

    result = tokenizer.decode(
        output_ids[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
