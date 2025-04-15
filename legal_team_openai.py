import streamlit as st
import os
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# ✅ Config
st.set_page_config(page_title="AI Legal Team Agents", page_icon="⚖️", layout="wide")

# ✅ Load API Key from secrets
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("OPENAI_API_KEY not found in Streamlit secrets.")
    st.stop()

# ✅ Model
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# ✅ Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = None

# ✅ Sidebar
with st.sidebar:
    st.header("📄 Upload Contract PDF")
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf"])

    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=500)
    overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100)

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            chunks = splitter.split_documents(pages)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

            st.session_state.vector_db = vectorstore
            st.session_state.doc_chunks = chunks

            st.success("✅ Document processed and indexed!")

# ✅ Multi-Agent Simulation
def get_agent_response(role: str, prompt: str):
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = st.session_state.vector_db.similarity_search(prompt, k=5)
    response = chain.run(input_documents=docs, question=prompt)

    return f"**{role} Response:**\n\n{response}"

def get_team_response(user_query):
    research = get_agent_response("Legal Advisor", f"Find legal precedents, cases, and regulations relevant to: {user_query}")
    analysis = get_agent_response("Contract Analyst", f"Analyze clauses, obligations, and ambiguities in: {user_query}")
    strategy = get_agent_response("Legal Strategist", f"Assess legal risks and give strategic advice for: {user_query}")

    final_prompt = f"""
    Based on the following:
    - Legal Advisor's input:\n{research}
    - Contract Analyst's input:\n{analysis}
    - Legal Strategist's input:\n{strategy}

    Summarize everything into a structured legal report with key insights and actionable recommendations.
    """

    team_lead_summary = get_agent_response("Team Lead", final_prompt)
    return research, analysis, strategy, team_lead_summary

# ✅ UI
if st.session_state.vector_db:
    st.header("🧠 Legal Analysis")

    analysis_type = st.selectbox("Choose Task:", [
        "Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"
    ])

    if analysis_type == "Custom Query":
        user_query = st.text_area("Enter your legal query:")
    else:
        preset_prompts = {
            "Contract Review": "Review this contract and summarize terms, obligations, and risks.",
            "Legal Research": "Find legal cases and precedents relevant to this contract.",
            "Risk Assessment": "Identify potential legal risks and areas of ambiguity.",
            "Compliance Check": "Check this contract for legal and regulatory compliance."
        }
        user_query = preset_prompts[analysis_type]

    if st.button("🔍 Analyze"):
        with st.spinner("Running legal department agents..."):
            res1, res2, res3, final = get_team_response(user_query)

            tabs = st.tabs(["Legal Advisor", "Contract Analyst", "Legal Strategist", "Final Report"])

            with tabs[0]:
                st.subheader("📚 Legal Advisor")
                st.markdown(res1)

            with tabs[1]:
                st.subheader("📑 Contract Analyst")
                st.markdown(res2)

            with tabs[2]:
                st.subheader("🛡 Legal Strategist")
                st.markdown(res3)

            with tabs[3]:
                st.subheader("🧾 Final Report (Team Lead)")
                st.markdown(final)
