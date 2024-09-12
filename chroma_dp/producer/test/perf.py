# generate performance test dataset for Chroma
import os.path
import tarfile
import uuid

import orjson as json
import json as json_std
from typing import Annotated, Optional, List, Dict, Any, Union, Tuple
import typer
from chromadb import GetResult, Where, WhereDocument
from chromadb.api.models import Collection
from chromadb.api.types import validate_where, validate_where_document
from numpy import ndarray, dtype, bool_

from chroma_dp import EmbeddableTextResource
from chroma_dp.chroma.chroma_import import chroma_import
from chroma_dp.utils.chroma import CDPUri, get_client_for_uri
import numpy as np
from numpy.typing import NDArray
from essential_generators import DocumentGenerator


def _generate_norm_dist(
    size: int, mean: float, std: float, is_int: bool = True
) -> NDArray[Any]:
    """
    Generate a normal distribution with specified mean and standard deviation,
    either as integers or floats, and clip to the range [0, 1000].
    """
    # Generate normal distribution
    normal_dist = np.random.normal(loc=mean, scale=std, size=int(size * 1.5))

    # Define 3-sigma range
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # Filter values within 3 standard deviations
    filtered_normal_dist = normal_dist[
        (normal_dist >= lower_bound) & (normal_dist <= upper_bound)
    ]

    # Clip values between 0 and 1000
    clipped_dist = np.clip(filtered_normal_dist, 0, 1000)
    clipped_dist = clipped_dist[:size]
    # Convert to integers if required
    if is_int:
        return clipped_dist.astype(int)
    else:
        return clipped_dist


def _get_ranges(
    normal_dist: NDArray[Any], std: float, mean: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate counts within 1, 2, and 3 sigma ranges for the given distribution.
    We use this for generating random
    """
    ranges = {
        "1-sigma": (mean - std, mean + std),
        "2-sigma": (mean - 2 * std, mean + 2 * std),
        "3-sigma": (mean - 3 * std, mean + 3 * std),
    }

    inclusive_counts = {
        k: (v, np.sum((normal_dist >= v[0]) & (normal_dist <= v[1])))
        for k, v in ranges.items()
    }
    exclusive_counts = {
        k: (v, np.sum((normal_dist > v[0]) & (normal_dist < v[1])))
        for k, v in ranges.items()
    }
    return inclusive_counts, exclusive_counts


def _eq_buckets_from_normal_dist(
    normal_dist: NDArray[Any], sample_size: int
) -> Dict[Union[int, float], int]:
    """
    Generate equal query buckets from a normal distribution
    """
    if sample_size > len(normal_dist):
        raise ValueError(
            f"Sample size {sample_size} is greater than the size of the distribution {len(normal_dist)}"
        )
    unique_elements, counts = np.unique(normal_dist, return_counts=True)
    repeated_elements = unique_elements[counts > 1]
    if len(repeated_elements) > 0:
        random_numbers = np.random.choice(
            repeated_elements,
            size=min(sample_size, len(repeated_elements)),
            replace=False,
        )
    else:
        random_numbers = np.random.choice(
            unique_elements, size=min(sample_size, len(unique_elements)), replace=False
        )
    eq_buckets = {k: np.sum(normal_dist == k) for k in random_numbers}
    return eq_buckets


def _eq_query_for_bucket(key: str, bucket: Union[int, float]) -> Where:
    """
    Generate equal queries for each bucket
    """
    return {f"{key}": bucket}


def _get_random_entries(
    normal_dist: NDArray[Any], sample_size: int
) -> List[Dict[str, Any]]:
    """
    We use this to generate random eq queries
    """
    if sample_size > len(normal_dist):
        raise ValueError(
            f"Sample size {sample_size} is greater than the size of the distribution {len(normal_dist)}"
        )
    return np.random.choice(np.unique(normal_dist), size=sample_size, replace=False)


gen = DocumentGenerator()


def _generate_random_string_words(size: int) -> List[str]:
    return [gen.gen_word() for _ in range(size)]


def _generate_random_sentences(size: int) -> List[str]:
    return [gen.gen_sentence() for _ in range(size)]


def _get_inclusive_range_query(key: str, range: Tuple[int, int]) -> Where:
    """
    We use this to generate range queries
    """
    return {"$and": [{f"{key}": {"$gte": range[0]}}, {f"{key}": {"$lte": range[0]}}]}


def _get_exclusive_range_query(key: str, range: Tuple[int, int]) -> Where:
    """
    We use this to generate range queries
    """
    return {"$and": [{f"{key}": {"$gt": range[0]}}, {f"{key}": {"$lt": range[0]}}]}


class NumpyEncoder(json_std.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert to native Python types
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert to a list
        return super(NumpyEncoder, self).default(obj)


def perf_dataset_cli(
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="The directory where the performance dataset will be saved",
    ),
    size: int = typer.Option(
        None,
        "--size",
        help="The size of the dataset to generate",
    ),
    random_docs: Optional[str] = typer.Option(
        None,
        "--random-docs",
        help="The number of random documents to generate",
    ),
    docs_file: Optional[str] = typer.Option(
        None,
        "--docs-file",
        help="The file containing the random documents",
    ),
    random_samples: Optional[int] = typer.Option(
        10,
        "--random-samples",
        "-r",
        help="The number of random samples to generate",
    ),
) -> None:
    dataset_metadata = []
    if not output_dir:
        raise ValueError("Output directory is required")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not size:
        raise ValueError("Size of the dataset is required")
    float_dist = _generate_norm_dist(size=size, mean=500.0, std=100.0, is_int=False)
    float_ranges = _get_ranges(float_dist, std=100.0, mean=500.0)
    # Example usage for integers
    int_dist = _generate_norm_dist(size=size, mean=500, std=100, is_int=True)
    int_ranges = _get_ranges(int_dist, std=100, mean=500)

    # generate range queries
    for k, v in float_ranges[0].items():
        dataset_metadata.append(
            {
                "id": f"inclusive_float_val_range_{k}_{v[0][0]}_{v[0][1]}",
                "tags": ["range", "float", "inclusive"],
                "query": _get_inclusive_range_query("float_val", v[0]),
                "count": v[1],
            }
        )
    for k, v in float_ranges[1].items():
        dataset_metadata.append(
            {
                "id": f"exclusive_float_val_range_{k}_{v[0][0]}_{v[0][1]}",
                "tags": ["range", "float", "exclusive"],
                "query": _get_exclusive_range_query("float_val", v[0]),
                "count": v[1],
            }
        )

    for k, v in int_ranges[0].items():
        dataset_metadata.append(
            {
                "id": f"inclusive_int_val_range_{k}_{v[0][0]}_{v[0][1]}",
                "tags": ["range", "int", "inclusive"],
                "query": _get_inclusive_range_query("int_val", v[0]),
                "count": v[1],
            }
        )
    for k, v in int_ranges[1].items():
        dataset_metadata.append(
            {
                "id": f"exclusive_int_val_range_{k}_{v[0][0]}_{v[0][1]}",
                "tags": ["range", "int", "exclusive"],
                "query": _get_exclusive_range_query("int_val", v[0]),
                "count": v[1],
            }
        )

    # generate equal queries
    int_eq_buckets = _eq_buckets_from_normal_dist(int_dist, random_samples or 10)
    for k, v in int_eq_buckets.items():
        dataset_metadata.append(
            {
                "id": f"eq_query_int_val_{k}",
                "tags": ["eq", "int"],
                "query": _eq_query_for_bucket("int_val", k),
                "count": v,
            }
        )
    float_eq_buckets = _eq_buckets_from_normal_dist(float_dist, random_samples or 10)
    for k, v in float_eq_buckets.items():
        dataset_metadata.append(
            {
                "id": f"eq_query_float_val_{k}",
                "tags": ["eq", "float"],
                "query": _eq_query_for_bucket("float_val", k),
                "count": v,
            }
        )

    docs = _generate_random_sentences(size)
    metadatas = []
    random_metadatas = np.random.choice(
        [i for i in range(size)], size=random_samples or 10, replace=False
    )
    ids = [str(uuid.uuid4()) for _ in range(size)]
    for i in range(size):
        num_metadatas = np.random.randint(1, 5)
        string_meta_keys = _generate_random_string_words(num_metadatas)
        string_meta_values = _generate_random_string_words(num_metadatas)
        _item_data = {k: v for k, v in zip(string_meta_keys, string_meta_values)}
        metadatas.append(_item_data)
        if i in random_metadatas:
            dataset_metadata.append(
                {
                    "id": f"random_meta_{i}",
                    "query": {
                        f"{list(metadatas[i].keys())[0]}": f"{list(metadatas[i].values())[0]}"
                    },
                    "tags": ["meta", "random", "eq"],
                    "count": 1,
                    "full_metadata": metadatas[i],
                }
            )
    with open(os.path.join(output_dir, "dataset.jsonl"), "w") as f:
        for i in range(size):
            metadata = metadatas[i]
            metadata.update({"int_val": int(int_dist[i])})
            metadata.update({"float_val": float_dist[i]})
            json_std.dump(
                EmbeddableTextResource(
                    id=ids[i],
                    text_chunk=docs[i],
                    embedding=[0.1, 0.2],
                    metadata=metadata,
                ).model_dump(),
                f,
                cls=NumpyEncoder,
            )
            f.write("\n")
    with open(os.path.join(output_dir, "queries.jsonl"), "w") as f:
        for query in dataset_metadata:
            json_std.dump(query, f, cls=NumpyEncoder)
            f.write("\n")

    # cdp import "file://testds-1/chroma-data/test" --create  --batch-size 10000
    chroma_import(
        uri=f"file://{os.path.join(output_dir, 'chroma-data', 'test')}",
        create=True,
        batch_size=10000,
        embedding_function=None,
        import_file=os.path.join(output_dir, "dataset.jsonl"),
        distance_function=None,
        max_threads=4,
        collection=None,
        meta_features=None,
    )

    with tarfile.open(
        os.path.join(output_dir, "..", f"{os.path.basename(output_dir)}.tar.gz"), "w:gz"
    ) as tar:
        tar.add(os.path.join(output_dir))
