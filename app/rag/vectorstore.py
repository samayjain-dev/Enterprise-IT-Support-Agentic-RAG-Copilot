import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# ---------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------

EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    model_name = (
        model_name
        or settings.embedding_model
        or "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()

    # Exact match
    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]

    # Flexible match
    if "all-minilm-l6-v2" in model_name.lower():
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name}'. "
        "Add its embedding dimension to EMBEDDING_DIMENSIONS."
    )


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        model_name = (
            settings.embedding_model
            or "sentence-transformers/all-MiniLM-L6-v2"
        )

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,

            model_kwargs={
                "device": "cpu",
            },

            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    return _embeddings


# ---------------------------------------------------------
# Pinecone
# ---------------------------------------------------------

def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("Pinecone API Key is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    index_names = [
        index["name"]
        for index in pc.list_indexes()
    ]

    # Check existing index
    if settings.pinecone_index_name in index_names:

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None,
        )

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        # Delete incompatible index
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            print(
                f"Deleting Pinecone index because dimension "
                f"changed from {current_dimension} "
                f"to {desired_dimension}"
            )

            pc.delete_index(
                name=settings.pinecone_index_name
            )

            while (
                settings.pinecone_index_name
                in [
                    index["name"]
                    for index in pc.list_indexes()
                ]
            ):
                time.sleep(1)

    # Create index if it doesn't exist
    index_names = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if settings.pinecone_index_name not in index_names:

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

    return pc.Index(
        settings.pinecone_index_name
    )


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )

def add_documents(chunks):
    store=get_vectorstore()
    return store.add_documents(chunks)

