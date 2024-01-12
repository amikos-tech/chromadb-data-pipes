from typing import Optional, Dict, Any, List

import chromadb
from chromadb import ClientAPI, GetResult
from chromadb.api.models.Collection import Collection
import urllib.parse

from pydantic import BaseModel, Field

from chroma_dp import ChromaDocument


def check_collection_exists(client: ClientAPI, collection_name: str) -> bool:
    """Checks if a collection exists in ChromaDB."""
    collections = client.list_collections()
    return collection_name in [collection.name for collection in collections]


def create_collection(client: ClientAPI, collection_name: str, if_not_exist: bool = False) -> Collection:
    """Creates a collection in ChromaDB."""
    if if_not_exist:
        return client.get_or_create_collection(collection_name)
    return client.create_collection(collection_name)


def get_collection(client: ClientAPI, collection_name: str) -> Collection:
    """Gets a collection in ChromaDB."""
    return client.get_collection(collection_name)


def read_large_data_in_chunks(collection: Collection, offset: int = 0, limit: int = 100) -> GetResult:
    """Reads large data in chunks from ChromaDB."""
    result = collection.get(
        limit=limit,
        offset=offset,
        include=["embeddings", "documents", "metadatas"])
    return result


from urllib.parse import urlparse, parse_qs


class CDPUri(BaseModel):
    auth: Optional[dict] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    collection: Optional[str] = None
    tenant: Optional[str] = None
    batch_size: Optional[int] = None
    limit: Optional[int] = Field(None,
                                 description="Limit of documents to export. Note: "
                                             "This parameter is only valid for chroma exports")
    offset: Optional[int] = None
    create_collection: Optional[bool] = False
    upsert: Optional[bool] = False

    @staticmethod
    def from_uri(uri: str) -> "CDPUri":
        parsed = urlparse(uri)
        user_info = parsed.username or None
        password_token = parsed.password or None
        if user_info == '__auth_token__':
            auth = {
                'type': 'token',
                'token': password_token,
                'header': 'AUTHORIZATION'
            }
        elif user_info == '__x_chroma_token__':
            auth = {
                'type': 'token',
                'token': password_token,
                'header': 'X-CHROMA-TOKEN'
            }
        elif user_info is not None:
            auth = {
                'type': 'basic',
                'username': user_info,
                'password': password_token
            }
        else:
            auth = None

        host = parsed.hostname or ''
        port = parsed.port or ''

        # Splitting the path into database and collection
        path_components = parsed.path.strip('/').split('/')
        database = path_components[0] if len(path_components) > 0 else ''
        collection = path_components[1] if len(path_components) > 1 else ''

        # Parsing query parameters
        query_params = parse_qs(parsed.query)
        tenant = query_params.get('tenant', [None])[0]
        batch_size = query_params.get('batch_size', [None])[0]
        _create_collection = query_params.get('create_collection', [None])[0]
        upsert = query_params.get('upsert', [None])[0]

        return CDPUri(
            auth=auth,
            host=host,
            port=port,
            database=database,
            collection=collection,
            tenant=tenant,
            batch_size=batch_size,
            create_collection=_create_collection,
            upsert=upsert
        )


def get_client_for_uri(uri: CDPUri) -> ClientAPI:
    """Gets a ChromaDB client for a given URI."""
    client = chromadb.HttpClient(
        host=uri.host,
        port=f"{uri.port}",
        database=uri.database or chromadb.api.DEFAULT_DATABASE,
        tenant=uri.tenant or chromadb.api.DEFAULT_TENANT,
        # TODO auth
        # settings=chromadb.Settings(
        #     chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        #     chroma_client_auth_credentials=parsed_uri.auth,
        #     chroma_client_auth_token_transport_header="X-CHROMA-TOKEN",
        # )
    )
    return client


def remap_features(in_dict: Dict[str, Any], doc_feature: str, embed_feature: str, meta_features: List[str],
                   id_feature: str) -> ChromaDocument:
    _doc = in_dict[doc_feature]
    _embed = in_dict[embed_feature] if embed_feature else None
    _meta = {k: in_dict[k] for k in meta_features} if meta_features else None
    _id = in_dict[id_feature] if id_feature else None
    return ChromaDocument(text_chunk=_doc, embedding=_embed, metadata=_meta, id=_id)
