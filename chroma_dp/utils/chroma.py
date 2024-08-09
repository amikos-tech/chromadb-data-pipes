import os
from enum import Enum
from typing import Optional, Dict, Any, List, cast
from urllib.parse import urlparse, parse_qs
import chromadb
from chromadb import ClientAPI, GetResult
from chromadb.api.models.Collection import Collection

from pydantic import BaseModel, Field

from chroma_dp import EmbeddableTextResource


def check_collection_exists(client: ClientAPI, collection_name: str) -> bool:
    """Checks if a collection exists in ChromaDB."""
    collections = client.list_collections()
    return collection_name in [collection.name for collection in collections]


def create_collection(
    client: ClientAPI, collection_name: str, if_not_exist: bool = False
) -> Collection:
    """Creates a collection in ChromaDB."""
    if if_not_exist:
        return client.get_or_create_collection(collection_name)
    return client.create_collection(collection_name)


def get_collection(client: ClientAPI, collection_name: str) -> Collection:
    """Gets a collection in ChromaDB."""
    return client.get_collection(collection_name)


def read_large_data_in_chunks(
    collection: Collection, offset: int = 0, limit: int = 100
) -> GetResult:
    """Reads large data in chunks from ChromaDB."""
    result = collection.get(
        limit=limit, offset=offset, include=["embeddings", "documents", "metadatas"]
    )
    return result


class DistanceFunction(str, Enum):
    l2 = "l2"
    ip = "ip"
    cosine = "cosine"


class CDPUri(BaseModel):
    auth: Optional[Dict[str, str]] = None
    host_or_path: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    collection: Optional[str] = None
    tenant: Optional[str] = None
    batch_size: Optional[int] = None
    is_local: Optional[bool] = False
    limit: Optional[int] = Field(
        None,
        description="Limit of documents to export. Note: "
        "This parameter is only valid for chroma exports",
    )
    offset: Optional[int] = None
    create_collection: Optional[bool] = False
    upsert: Optional[bool] = False
    distance_function: Optional[DistanceFunction] = None

    @staticmethod
    def from_uri(uri: str) -> "CDPUri":
        parsed = urlparse(uri)
        if parsed.scheme not in ["http", "https", "file"]:
            raise ValueError(
                f"Unsupported scheme: {parsed.scheme}. Must be 'http', 'https' or 'file'."
            )

        is_local = False
        if parsed.scheme == "file":
            is_local = True
        user_info = parsed.username or None
        password_token = parsed.password or None
        if user_info == "__auth_token__":
            auth = {
                "type": "token",
                "token": password_token,
                "header": "Authorization",
            }
        elif user_info == "__x_chroma_token__":
            auth = {
                "type": "token",
                "token": password_token,
                "header": "X-Chroma-Token",
            }
        elif user_info is not None:
            auth = {
                "type": "basic",
                "username": user_info,
                "password": password_token,
            }
        else:
            auth = None

        host_or_path = parsed.hostname or ""
        port = parsed.port or "8000"

        # Splitting the path into database and collection
        collection = parsed.path.split("/")[-1]
        if is_local:
            host_or_path = host_or_path + parsed.path[: parsed.path.index(collection)]
        # Parsing query parameters
        query_params = parse_qs(parsed.query)
        database = query_params.get("database", [None])[0]
        tenant = query_params.get("tenant", [None])[0]
        batch_size = query_params.get("batch_size", [None])[0]
        _create_collection = query_params.get("create_collection", [None])[0]
        upsert = query_params.get("upsert", [None])[0]
        limit = query_params.get("limit", [None])[0]
        offset = query_params.get("offset", [None])[0]
        distance_function = query_params.get("df", [None])[0]

        return CDPUri(
            auth=auth,
            host_or_path=host_or_path,
            is_local=is_local,
            port=port,
            database=database,
            collection=collection,
            tenant=tenant,
            batch_size=batch_size,
            create_collection=_create_collection,
            upsert=upsert,
            limit=limit,
            offset=offset,
            distance_function=DistanceFunction[distance_function]
            if distance_function
            else None,
        )


def _get_auth_settings_for_uri(uri: CDPUri) -> chromadb.Settings:
    """Gets the authentication settings for a given URI."""
    if uri.auth is not None:
        if uri.auth["type"] == "basic":
            # provider = "chromadb.auth.basic_authn.BasicAuthClientProvider" if tuple(map(int, chromadb.__version__.split("."))) < (0, 5, 0) else ""
            # TODO support < 0.5.0?
            return chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                chroma_client_auth_credentials=f"{uri.auth['username']}:{uri.auth['password']}",
            )
        if uri.auth["type"] == "token":
            return chromadb.Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials=uri.auth["token"],
                chroma_auth_token_transport_header=uri.auth["header"],
            )
        raise ValueError(f"Unsupported auth type: {uri.auth['type']}")
    if os.environ.get("CHROMA_BASIC_AUTH"):
        return chromadb.Settings(
            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
            chroma_client_auth_credentials=os.environ["CHROMA_BASIC_AUTH"],
        )
    if os.environ.get("CHROMA_TOKEN_AUTH"):
        return chromadb.Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=os.environ["CHROMA_TOKEN_AUTH"],
            chroma_auth_token_transport_header="Authorization",
        )
    if os.environ.get("CHROMA_XTOKEN_AUTH"):
        return chromadb.Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=os.environ["CHROMA_XTOKEN_AUTH"],
            chroma_auth_token_transport_header="X-Chroma-Token",
        )
    return chromadb.Settings()


def get_client_for_uri(uri: CDPUri) -> ClientAPI:
    """Gets a ChromaDB client for a given URI."""
    if uri.is_local:
        client = chromadb.PersistentClient(
            path=uri.host_or_path,
            database=uri.database or chromadb.api.DEFAULT_DATABASE,
            tenant=uri.tenant or chromadb.api.DEFAULT_TENANT,
        )
    else:
        client = chromadb.HttpClient(
            host=uri.host_or_path,
            port=uri.port,
            database=uri.database or chromadb.api.DEFAULT_DATABASE,
            tenant=uri.tenant or chromadb.api.DEFAULT_TENANT,
            settings=_get_auth_settings_for_uri(
                uri
            ),  # TODO this is not optimal if we want to configure extra settings
        )

    return client


def get_default_metadata(in_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempts to get default metadata from a dictionary (of EmbeddableTextResource) or returns None."""
    if "metadata" in in_dict:
        return cast(Dict[str, Any], in_dict["metadata"])
    return None


def remap_features(
    in_dict: Dict[str, Any],
    doc_feature: str,
    embed_feature: str,
    id_feature: str,
    meta_features: Optional[List[str]] = None,
) -> EmbeddableTextResource:
    # if standard metadata is present and no specific metadata is provided use the standard metadata
    if "metadata" in in_dict and not meta_features:
        # print("in_dict", in_dict)
        _meta = get_default_metadata(in_dict)
    elif "metadata" in in_dict and meta_features:
        _meta = {k: in_dict["metadata"][k] for k in meta_features}
    elif "metadata" not in in_dict and meta_features:
        _meta = {k: in_dict[k] for k in meta_features}
    else:
        _meta = None
    _doc = in_dict[doc_feature]
    _embed = in_dict[embed_feature] if embed_feature else None
    _id = in_dict[id_feature] if id_feature else None
    return EmbeddableTextResource(
        text_chunk=_doc, embedding=_embed, metadata=_meta, id=_id
    )
