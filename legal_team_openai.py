import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="AI Legal Team Agents (OpenAI)", page_icon="⚖️", layout="wide")

# Core libraries
import os
import tempfile
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.document import DocumentChunking

# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = ChromaDb(
        collection="law",
        path="/tmp/chromadb",  # works on Render
        persistent_client=True,
        embedder=OpenAIEmbedder()
    )

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Sidebar
with st.sidebar:
    st.header("Configuration")

    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("OpenAI key loaded from secrets!")
    else:
        st.error("❌ OPENAI_API_KEY not found in secrets.")

    chunk_size_in = st.number_input("Chunk Size", min_value=100, max_value=5000, value=500)
    overlap_in = st.number_input("Overlap", min_value=0, max_value=1000, value=100)

    st.header("📄 Upload Contract PDF")
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    st.session_state.knowledge_base = PDFKnowledgeBase(
                        path=temp_path,
                        vector_db=st.session_state.vector_db,
                        reader=PDFReader(),
                        chunking_strategy=DocumentChunking(chunk_size=chunk_size_in, overlap=overlap_in)
                    )

                    st.session_state.knowledge_base.load(recreate=True, upsert=True)
                    st.session_state.processed_files.add(uploaded_file.name)

                    st.success("✅ Document indexed into knowledge base!")

                except Exception as e:
                    st.error(f"Error: {e}")

# Agents
if st.session_state.knowledge_base:
    legal_researcher = Agent(
        name="LegalAdvisor",
        model=OpenAIChat(model="gpt-4"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Finds and cites legal precedents.",
        instructions=[
            "Extract all relevant data from the knowledge base.",
            "Use DuckDuckGo if needed for extra references.",
            "Always cite sources."
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    contract_analyst = Agent(
        name="ContractAnalyst",
        model=OpenAIChat(model="gpt-4"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Analyzes contracts for risks and obligations.",
        instructions=[
            "Identify key terms, obligations, and any ambiguous clauses.",
            "Reference document sections."
        ],
        show_tool_calls=True,
        markdown=True
    )

    legal_strategist = Agent(
        name="LegalStrategist",
        model=OpenAIChat(model="gpt-4"),
        knowledge=st.session_state.knowledge_base,
        search_knowledge=True,
        description="Suggests strategy & compliance guidance.",
        instructions=[
            "Assess for legal risks and compliance issues.",
            "Recommend steps to strengthen the contract."
        ],
        show_tool_calls=True,
        markdown=True
    )

    team_lead = Agent(
        name="TeamLead",
        model=OpenAIChat(model="gpt-4"),
        description="Summarizes all agent outputs into a full report.",
        instructions=[
            "Merge findings from LegalAdvisor, ContractAnalyst, and LegalStrategist.",
            "Ensure references and recommendations are clear."
        ],
        show_tool_calls=True,
        markdown=True
    )

    def get_team_response(query):
        res1 = legal_researcher.run(query)
        res2 = contract_analyst.run(query)
        res3 = legal_strategist.run(query)

        final = team_lead.run(
            f"""Summarize and merge insights:
            LegalAdvisor: {res1}
            ContractAnalyst: {res2}
            LegalStrategist: {res3}

            Output a professional legal report.
            """
        )
        return final

    # UI
    st.header("🔍 Legal Analysis")
    analysis_type = st.selectbox("Choose task:", [
        "Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"
    ])

    query = ""
    if analysis_type == "Custom Query":
        query = st.text_area("Enter your legal question:")
    else:
        query_map = {
            "Contract Review": "Review this contract and summarize terms, obligations, and risks.",
            "Legal Research": "Find legal cases and precedents relevant to this contract.",
            "Risk Assessment": "Identify potential legal risks and areas of ambiguity.",
            "Compliance Check": "Check this contract for legal and regulatory compliance."
        }
        query = query_map[analysis_type]

    if st.button("🧠 Analyze"):
        with st.spinner("Running multi-agent analysis..."):
            response = get_team_response(query)

            tabs = st.tabs(["Full Report", "Key Points", "Recommendations"])

            with tabs[0]:
                st.subheader("📑 Full Legal Analysis")
                st.markdown(response.content)

            with tabs[1]:
                summary = team_lead.run(f"Summarize key points:\n{response.content}")
                st.subheader("📌 Key Points")
                st.markdown(summary.content)

            with tabs[2]:
                recommendations = team_lead.run(f"Give legal recommendations:\n{response.content}")
                st.subheader("📋 Recommendations")
                st.markdown(recommendations.content)
