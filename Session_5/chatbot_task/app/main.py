import streamlit as st
import asyncio
import logging
import json
import os
import uuid
from datetime import datetime
from src.chatbot import CustomChatBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

INDEX_DATA = os.environ.get("INDEX_DATA", "0")
PULL_EMBEDDING_MODEL = os.environ.get("PULL_EMBEDDING_MODEL", "0")
HISTORY_FILE = "/app/chat_history.json"

# --- CONFIG & DESIGN ---
st.set_page_config(page_title="Lern-Chatbot", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {background-color: #1e1e24;}
    .stChatInput input {border-radius: 20px !important;}
    h1, h2, h3 {color: #00e5ff !important;}
</style>
""", unsafe_allow_html=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return {}

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["all_chats"], f, ensure_ascii=False, indent=2)

def init_chat_session():
    new_id = str(uuid.uuid4())
    st.session_state["all_chats"][new_id] = {
        "title": "Neuer Chat",
        "messages": [{"role": "assistant", "content": "Hi! Worüber sprechen wir heute?"}],
        "timestamp": datetime.now().isoformat()
    }
    st.session_state["current_chat_id"] = new_id
    save_history()

# --- APP START ---
if "bot" not in st.session_state:
    st.session_state["bot"] = CustomChatBot(index_data=bool(int(INDEX_DATA)), pull_embedding_model=bool(int(PULL_EMBEDDING_MODEL)))

if "all_chats" not in st.session_state: st.session_state["all_chats"] = load_history()

if "current_chat_id" not in st.session_state or st.session_state["current_chat_id"] not in st.session_state["all_chats"]:
    if st.session_state["all_chats"]:
        sorted_chats = sorted(st.session_state["all_chats"].items(), key=lambda x: x[1]["timestamp"], reverse=True)
        st.session_state["current_chat_id"] = sorted_chats[0][0]
    else: init_chat_session()

current_msgs = st.session_state["all_chats"][st.session_state["current_chat_id"]]["messages"]

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗂️ Chats")
    if st.button("➕ Neuer Chat", use_container_width=True):
        init_chat_session()
        st.rerun()
    st.divider()
    
    sorted_chats = sorted(st.session_state["all_chats"].items(), key=lambda x: x[1]["timestamp"], reverse=True)
    for cid, cdata in sorted_chats[:3]:
        c1, c2 = st.columns([0.8, 0.2])
        lbl = f"📂 {cdata['title']}" if cid != st.session_state["current_chat_id"] else f"👉 **{cdata['title']}**"
        if c1.button(lbl, key=cid):
            st.session_state["current_chat_id"] = cid
            st.rerun()
        if c2.button("🗑️", key=f"d_{cid}"):
            del st.session_state["all_chats"][cid]
            save_history()
            st.rerun()

    st.divider()
    st.subheader("🛠️ Tools")
    
    uploaded_file = st.file_uploader("Datei lernen", type=["pdf", "docx", "txt"])
    if uploaded_file and st.button("Einlesen"):
        with st.spinner("Lese..."):
            path = f"/app/temp_{uploaded_file.name}"
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
            ok, msg = st.session_state["bot"].ingest_new_file(path, uploaded_file.name.split(".")[-1].lower())
            if ok: st.success(msg)
            else: st.error(msg)

    use_web = st.toggle("🌐 Web-Suche", value=False)
    
    # --- NEUE STIL AUSWAHL ---
    st.write("---")
    st.subheader("🎭 Antwort-Stil")
    mode = st.radio("Wähle dein Niveau:", ["Fachexperte 🧐", "Mittel ⚖️", "Fachidiot 👶"])
    
    # Mapping der Namen auf interne IDs
    pmap = {
        "Fachexperte 🧐": "expert",
        "Mittel ⚖️": "intermediate",
        "Fachidiot 👶": "beginner"
    }
    
    if st.session_state.get("last_mode") != mode:
        st.session_state["bot"].set_persona(pmap[mode])
        st.session_state["last_mode"] = mode
        st.success(f"Modus gewechselt: {mode}")

    # QUIZ
    st.write("---")
    if st.button("🎲 Quiz starten"):
        with st.spinner("Erstelle Quiz..."):
            topic = "General AI"
            bot_msgs = [m["content"] for m in current_msgs if m["role"] == "assistant"]
            if bot_msgs: topic = bot_msgs[-1]
            
            quiz_text = st.session_state["bot"].generate_quiz(topic)
            
            st.session_state["all_chats"][st.session_state["current_chat_id"]]["messages"].append({
                "role": "assistant", 
                "content": f"**QUIZ TIME!** 🎲\n\n{quiz_text}"
            })
            save_history()
            st.rerun()

    if st.button("📄 PDF Export"):
        p = st.session_state["bot"].save_chat_to_pdf(current_msgs)
        if p:
            with open(p, "rb") as f: st.download_button("⬇️ Download", f, file_name="Chat.pdf")

# --- MAIN CHAT ---
st.header("Dein Lern-Assistent 🤖")

bot_avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
user_avatar = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"

for msg in current_msgs:
    av = bot_avatar if msg["role"] == "assistant" else user_avatar
    with st.chat_message(msg["role"], avatar=av):
        st.write(msg["content"])

if inp := st.chat_input("Frage..."):
    st.session_state["all_chats"][st.session_state["current_chat_id"]]["messages"].append({"role": "user", "content": inp})
    with st.chat_message("user", avatar=user_avatar):
        st.write(inp)
    
    cc = st.session_state["all_chats"][st.session_state["current_chat_id"]]
    if len(cc["messages"]) <= 3: cc["title"] = " ".join(inp.split()[:4])
    save_history()

    async def run_bot():
        box = st.empty()
        full = ""
        hist = st.session_state["all_chats"][st.session_state["current_chat_id"]]["messages"]
        async for chunk in st.session_state["bot"].astream(inp, history=hist, use_web=use_web):
            if chunk:
                full += chunk
                box.markdown(full + "▌")
            box.markdown(full)
        return full

    with st.chat_message("assistant", avatar=bot_avatar):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ans = loop.run_until_complete(run_bot())
        st.session_state["all_chats"][st.session_state["current_chat_id"]]["messages"].append({"role": "assistant", "content": ans})
        save_history()