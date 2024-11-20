import os
from pprint import pprint
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
retriever = vectorstore.as_retriever()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise""",
            ),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )
    | llm
    | StrOutputParser()
)

retrieval_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
            ),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )
    | llm
    | JsonOutputParser()
)

rag_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise""",
            ),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )
    | llm
    | StrOutputParser()
)

hallucination_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation.""",
            ),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )
    | llm
    | JsonOutputParser()
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    generate_count: int
    web_search: str
    web_search_count: int
    grounded: str
    failed: str
    documents: List[str]


def init(state):
    return {"web_search_count": 0, "generate_count": 0}


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generate_count": state["generate_count"] + 1,
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def grade_hallucination(state):
    """
    Determines whether the generation is hallucinated
    """
    print("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "grounded": score["score"],
    }


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = None
    if "documents" in state:
        documents = state["documents"]
    docs = tavily.search(query=question)["results"]

    for doc in docs:
        if documents is not None:
            documents.append(
                Document(
                    page_content=doc["content"],
                    metadata={"source": doc["url"], "title": doc["title"]},
                )
            )
        else:
            documents = [
                Document(
                    page_content=doc["content"],
                    metadata={"source": doc["url"], "title": doc["title"]},
                )
            ]
    return {
        "documents": documents,
        "question": question,
        "web_search_count": state["web_search_count"] + 1,
    }


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    if state["web_search_count"] == 1:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, NOT RELEVANT---"
        )
        return "not_relevant"
    if state["web_search"].lower() == "yes":
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    if state["generate_count"] == 2:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, NOT SUPPORTED---")
        return "not_supported"
    if state["grounded"] == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "supported"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate"


workflow = StateGraph(GraphState)

workflow.add_node("init", init)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("grade_hallucination", grade_hallucination)  # grade hallucination
# Build graph
workflow.set_entry_point("init")
workflow.add_edge("init", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
        "not_relevant": END,
    },
)
workflow.add_edge("websearch", "grade_documents")
workflow.add_edge("generate", "grade_hallucination")
workflow.add_conditional_edges(
    "grade_hallucination",
    grade_generation_v_documents_and_question,
    {
        "generate": "generate",
        "not_supported": END,
        "supported": END,
    },
)
app = workflow.compile()

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)

# Streamlit 앱 UI
st.title("Research Assistant powered by OpenAI")

input_topic = st.text_input(
    ":female-scientist: Enter a topic",
    value="Superfast Llama 3 inference on Groq Cloud",
)

generate_report = st.button("Generate Report")

if generate_report:
    with st.spinner("Generating Report"):
        inputs = {"question": input_topic}
        output = app.invoke(inputs)
        final_report = ""
        if output["generate_count"] == 2:
            final_report += "failed: hallucination"
        elif output["web_search_count"] == 1:
            final_report += "failed: web search"
        else:
            final_report += "# Answer\n\n"
            final_report += output["generation"]
            final_report += "\n\n# References\n"
            sources = {
                d.metadata["source"]: d.metadata["title"] for d in output["documents"]
            }
            for source, title in sources.items():
                final_report += f" - {title}: {source}\n"
        st.markdown(final_report)

st.sidebar.markdown("---")
if st.sidebar.button("Restart"):
    st.session_state.clear()
    st.experimental_rerun()
