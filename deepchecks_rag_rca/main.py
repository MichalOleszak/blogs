import os
import time
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ---- BEIR (dataset) ----
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ---- LlamaIndex ----
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

# ---- Deepchecks ----
from deepchecks_llm_client.client import DeepchecksLLMClient
from deepchecks_llm_client.data_types import (
    ApplicationType,
    EnvType,
    LogInteraction,
)

# TODO: Set these env vars
# os.environ["DEEPCHECKS_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""


PROMPTS = {
    "SYSTEM_PROMPT": {
        "BASELINE": (
            "You are a knowledgeable assistant. "
            "Answer user questions."
        ),
        "STRICT": (
            "You are a careful assistant for Retrieval-Augmented QA.\n\n"
            "Rules:\n"
            "1) Use ONLY the provided context to answer.\n"
            "2) Always include direct quotes from the context to support your answer.\n"
            "3) If the context does not fully support an answer, reply exactly: \"I don't know.\"\n"
            "4) Never add outside knowledge or speculation."
        ),
    },
    "USER_PREAMBLE": {
        "BASELINE": (
            "Answer briefly."
        ),
        "STRICT": (
            "Follow the rules strictly:\n"
            "- Quote exact phrases from the context to support your answer.\n"
            "- If the answer is not supported, reply: \"I don't know.\"\n"
            "- Do not use information not present in the context.\n\n"
            "Answer:"
        ),
    }
}



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_beir(dataset: str, split: str = "test") -> Tuple[Dict, Dict, Dict]:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(os.getcwd(), "beir_datasets")
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


def corpus_to_documents(corpus: Dict[str, Dict]) -> List[Document]:
    docs: List[Document] = []
    for doc_id, rec in corpus.items():
        title = rec.get("title") or ""
        text = rec.get("text") or ""
        content = f"{title}\n{text}" if title and text else title or text
        if not content.strip():
            continue
        docs.append(Document(text=content, doc_id=doc_id, metadata={"beir_id": doc_id}))
    return docs


def build_index(
    documents: List[Document],
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    chunk_size_limit: int = 256
) -> VectorStoreIndex:
    # Configure global Settings (keeps code simpler)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.llm = None  # we set LLM at query time
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.chunk_size_limit = chunk_size_limit
    index = VectorStoreIndex.from_documents(documents)
    return index


def make_query_engine(
    index: VectorStoreIndex,
    k: int,
    strict_prompt: bool,
    use_reranker: bool,
    openai_model: str,
) -> RetrieverQueryEngine:
    
    Settings.llm = LlamaOpenAI(model=openai_model, temperature=0.0)

    # System + user prompts via LlamaIndex response_mode prompt wrappers
    if strict_prompt:
        system_prompt = PROMPTS.get("SYSTEM_PROMPT").get("STRICT")
        user_preamble = PROMPTS.get("USER_PREAMBLE").get("STRICT")
    else:
        system_prompt = PROMPTS.get("SYSTEM_PROMPT").get("BASELINE")
        user_preamble = PROMPTS.get("USER_PREAMBLE").get("BASELINE")

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=(k * 5 if use_reranker else k),
    )

    postprocs = []
    if use_reranker:
        postprocs.append(
            SentenceTransformerRerank(
                model="BAAI/bge-reranker-base",
                top_n=k,
            )
        )

    qe = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=postprocs,
    )

    # Convenience wrapper: add rules to the user query, capture "full_prompt",
    # and return (answer, full_prompt, raw_response)
    def ask(question: str):
        user_msg = f"{user_preamble}\n\nQuestion: {question}"
        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_msg}"
        response = qe.query(str_or_query_bundle=user_msg)
        return str(response), full_prompt, response

    qe.ask = ask  # type: ignore[attr-defined]
    return qe


def ensure_app_and_versions(dc: DeepchecksLLMClient, app_name: str, baseline_version: str, improved_version: str) -> None:
    try:
        dc.create_application(app_name, app_type=ApplicationType.QA)
    except Exception:
        pass
    for name, desc in [
        (baseline_version, "Baseline LlamaIndex RAG"),
        (improved_version, "Strict prompt + cross-encoder reranker"),
    ]:
        try:
            dc.create_app_version(app_name=app_name, version_name=name, description=desc)
        except Exception:
            pass


def to_interactions(
    qe: RetrieverQueryEngine,
    queries: Dict[str, str],
    k: int,
    max_queries: int = None,
) -> List[LogInteraction]:
    items = list(queries.items())
    if max_queries:
        items = items[:max_queries]

    interactions: List[LogInteraction] = []
    for qid, question in tqdm(items, desc="Answering"):
        try:
            started = int(time.time())
            answer, full_prompt, resp = qe.ask(question)
    
            # Gather retrieved chunks for Deepchecks
            retrieved_chunks: List[str] = []
            try:
                # LlamaIndex returns response.source_nodes with node.text and scores
                for node in resp.source_nodes[:k]:
                    retrieved_chunks.append(node.node.get_content(metadata_mode="all"))
            except Exception:
                pass
    
            finished = int(time.time())
    
            interactions.append(
                LogInteraction(
                    user_interaction_id=qid,
                    input=question,
                    output=answer,
                    full_prompt=full_prompt,
                    information_retrieval=retrieved_chunks,
                    started_at=started,
                    finished_at=finished,
                    metadata={},
                )
            )
        except Exception as e:
            print(f"Error: {e}")
    return interactions


def upload_batch(dc: DeepchecksLLMClient, app_name: str, version_name: str, interactions: List[LogInteraction]) -> None:
    dc.log_batch_interactions(
        app_name=app_name,
        version_name=version_name,
        env_type=EnvType.EVAL,
        interactions=interactions,
    )


def check_env():
    for env_var in ["DEEPCHECKS_API_KEY", "OPENAI_API_KEY"]:
        if os.getenv(env_var) is None:
            raise Exception(f"Environment variable '{env_var}' is not set.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-name", type=str, default="RAG-RCA")
    parser.add_argument("--baseline-version", type=str, default="baseline")
    parser.add_argument("--improved-version", type=str, default="strict+rerank")
    parser.add_argument("--dataset", type=str, default="scifact", help="e.g., scifact | fiqa | hotpotqa")
    parser.add_argument("--max-queries", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--openai-model", type=str, default="gpt-5-nano")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--n-queries", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    check_env()
    APP_NAME = args.app_name + "-" + args.dataset

    # Deepchecks client
    dc = DeepchecksLLMClient(api_token=os.environ.get("DEEPCHECKS_API_KEY"))
    ensure_app_and_versions(dc, APP_NAME, args.baseline_version, args.improved_version)
    print("✅ Deepchecks client and project set up successfully")
    
    # Load data
    corpus, queries, qrels = load_beir(args.dataset, split="test")
    documents = corpus_to_documents(corpus)
    if len(documents) > args.n_docs:
        documents = random.sample(documents, args.n_docs)
    if len(queries) > args.n_queries:
        sampled_keys = random.sample(list(queries.keys()), args.n_queries)
        queries = {k: queries[k] for k in sampled_keys}
    print(f"✅ Data loaded successfully:\n- {len(documents)} documents\n- {len(queries)} queries")

    # Build LlamaIndex vector index
    index = build_index(
        documents=documents,
        embed_model_name=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Build query engines
    baseline_qe = make_query_engine(
        index=index,
        k=args.k,
        strict_prompt=False,
        use_reranker=False,
        openai_model=args.openai_model,
    )
    improved_qe = make_query_engine(
        index=index,
        k=args.k,
        strict_prompt=True,
        use_reranker=True,
        openai_model=args.openai_model,
    )
    print("✅ Vector index and query engines built successfully")

    # Run baseline
    baseline_interactions = to_interactions(
        qe=baseline_qe,
        queries=queries,
        k=args.k,
        max_queries=args.max_queries,
    )
    upload_batch(dc,APP_NAME, args.baseline_version, baseline_interactions)
    print("✅ Baseline interactions uploaded to Deepchecks")

    # Run improved
    improved_interactions = to_interactions(
        qe=improved_qe,
        queries=queries,
        k=args.k,
        max_queries=args.max_queries,
    )
    upload_batch(dc, APP_NAME, args.improved_version, improved_interactions)
    print("✅ Improved interactions uploaded to Deepchecks")


if __name__ == "__main__":
    main()
    print("🎉 Done")
