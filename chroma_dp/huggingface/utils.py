import datasets


def _infer_hf_type(value):
    """
    Infers the Hugging Face data type from a Python type.

    Args:
        value: The value to infer the data type from.

    Returns:
        The inferred Hugging Face data type.
    """
    if isinstance(value, bool):
        return datasets.Value("bool")
    elif isinstance(value, int):
        return datasets.Value("int32")
    elif isinstance(value, float):
        return datasets.Value("float32")
    elif isinstance(value, str):
        return datasets.Value("string")
    elif isinstance(value, list):
        if all(isinstance(elem, int) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='int32'))
        elif all(isinstance(elem, float) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='float32'))
        elif all(isinstance(elem, str) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='string'))
        else:
            raise ValueError("Unsupported list type")
    else:
        raise ValueError("Unsupported type")
