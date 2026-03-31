"""RAG chain construction with history-aware retrieval and conversation memory.

This module wires together:

1. A **history-aware retriever** — rewrites the user's follow-up questions
   into standalone questions using the conversation history before hitting the
   vector store.
2. A **question-answer chain** — injects the retrieved document context and
   the chat history into a prompt and calls the LLM.
3. :class:`~langchain_core.runnables.history.RunnableWithMessageHistory` —
   automatically reads/writes the per-session history managed by
   :class:`~rag.memory.ConversationMemory`.
"""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from rag.memory import ConversationMemory

_CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question — just "
    "reformulate it if needed and otherwise return it as-is."
)

_QA_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise — three sentences maximum.\n\n"
    "{context}"
)


def build_rag_chain(
    llm,
    retriever: VectorStoreRetriever,
    memory: ConversationMemory,
) -> Runnable:
    """Build a conversational RAG chain with persistent per-session memory.

    The returned chain is a
    :class:`~langchain_core.runnables.history.RunnableWithMessageHistory`
    that expects::

        chain.invoke(
            {"input": "<user question>"},
            config={"configurable": {"session_id": "<session id>"}},
        )

    and returns a dict with an ``"answer"`` key.

    Args:
        llm: A LangChain-compatible chat model (e.g. ``ChatOpenAI``).
        retriever: A vector-store retriever used to fetch relevant chunks.
        memory: A :class:`~rag.memory.ConversationMemory` instance that stores
            per-session chat histories.

    Returns:
        A :class:`~langchain_core.runnables.Runnable` ready to be invoked.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        memory.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
