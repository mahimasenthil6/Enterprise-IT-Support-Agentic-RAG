
import logging
from typing import Literal

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


# --------------------------------------------------
# 1. Initialize Groq LLM
# --------------------------------------------------

def llm():
    global _llm

    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")

        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    return _llm


# --------------------------------------------------
# 2. Initialize Tavily Web Search
# --------------------------------------------------

def web_search_tool():
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


# --------------------------------------------------
# 3. Maintain Agent Execution Trace
# --------------------------------------------------

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# --------------------------------------------------
# 4. Route User Question
# --------------------------------------------------

def route_question(state: AgentState):

    router = llm().with_structured_output(
        RouteDecision,
        method="json_schema",
        strict=True,
    )

    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.

Use kb for questions about company IT policies, VPN, password reset,
MFA, laptop setup, software access, security, email, devices,
troubleshooting, or technology support.

Use direct only for greetings, thanks, or casual chat
that needs no company knowledge.

Question: {state['question']}

Return a structured response with route equal to kb or direct.
""")

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}",
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:

    return (
        "retrieve_kb"
        if state["source_used"] == "kb"
        else "direct_answer"
    )


# --------------------------------------------------
# 5. Retrieve Documents from Private Knowledge Base
# --------------------------------------------------

def retrieve_kb(state: AgentState):

    docs = get_retriever().invoke(state["current_query"])

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


# --------------------------------------------------
# 6. Grade Knowledge Base Evidence
# --------------------------------------------------

def grade_kb(state: AgentState):

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema",
        strict=True,
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Question: {state['question']}

Private company KB evidence:
{context}

Return good only if the evidence is sufficient to answer
confidently and specifically.

Otherwise return weak.
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:

    return (
        "generate_from_kb"
        if state["kb_grade"] == "good"
        else "search_web"
    )


# --------------------------------------------------
# 7. Search External Web Using Tavily
# --------------------------------------------------

def search_web(state: AgentState):

    result = web_search_tool().invoke(
        {"query": state["current_query"]}
    )

    lines = []
    citations = []

    if isinstance(result, dict):

        if result.get("answer"):
            lines.append(
                "Search answer: " + result["answer"]
            )

        for item in result.get("results", []):

            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            citations.append({
                "title": title or url,
                "url": url,
                "type": "web",
            })

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }


# --------------------------------------------------
# 8. Grade Web Evidence
# --------------------------------------------------

def grade_web(state: AgentState):

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema",
        strict=True,
    )

    grade = grader.invoke(f"""
Question: {state['question']}

Web evidence:
{state['web_results']}

Return good if the evidence is sufficient and directly relevant.
Otherwise return weak.
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# --------------------------------------------------
# 9. Rewrite Question for Better Retrieval
# --------------------------------------------------

def rewrite_query(state: AgentState):

    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private
knowledge retrieval and vendor web search.

Preserve the original intent.
Add useful technical keywords.
Do not answer the question.
Return only the rewritten query.

Question: {state['question']}
""").content.strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }


# --------------------------------------------------
# 10. Generate Answer from Private Knowledge Base
# --------------------------------------------------

def generate_from_kb(state: AgentState):

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company KB below.

Be concise, actionable, and safe.
If steps are present, present them clearly.
Do not invent policy details.
Mention that the answer is based on the company's
private knowledge base.

Question: {state['question']}

Private KB:
{context}
""").content

    citations = []
    seen = set()

    for d in state["kb_docs"]:

        src = d.metadata.get("source", "Private KB")

        if src not in seen:
            seen.add(src)

            citations.append({
                "title": src.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


# --------------------------------------------------
# 11. Generate Answer from Web Search
# --------------------------------------------------

def generate_from_web(state: AgentState):

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

The private company KB was insufficient.

Answer ONLY from the web evidence below.

Clearly say this is external web information
and may need IT validation before changing
company-managed systems.

Question: {state['question']}

Web evidence:
{state['web_results']}
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


# --------------------------------------------------
# 12. Direct Answer for Greetings and Casual Chat
# --------------------------------------------------

def direct_answer(state: AgentState):

    answer = llm().invoke(
        f"Respond briefly and naturally to: {state['question']}"
    ).content

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


# --------------------------------------------------
# 13. Handle Insufficient Evidence
# --------------------------------------------------

def insufficient(state: AgentState):

    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to answer "
            "confidently. Please contact the IT help desk or "
            "provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


# --------------------------------------------------
# 14. Build LangGraph Workflow
# --------------------------------------------------

def build_graph():

    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()


# --------------------------------------------------
# 15. Compile Agent Graph
# --------------------------------------------------

agent_graph = build_graph()


# --------------------------------------------------
# 16. Ask Function - Entry Point
# --------------------------------------------------

def ask(question: str):

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)
