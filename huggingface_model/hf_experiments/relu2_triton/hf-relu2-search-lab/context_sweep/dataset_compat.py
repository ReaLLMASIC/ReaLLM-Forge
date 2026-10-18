"""Read serialized List features in the pinned datasets 3.6 evaluation worker.

Only feature metadata is translated. A variable-length List becomes the native
3.6 singleton-list representation, including lists of structs. Mapping it to
Sequence unconditionally would incorrectly transpose lists of structs.
No package, dataset row, model or training environment is changed on disk.
"""


def legacy_list_metadata(obj):
    if isinstance(obj, list):
        return [legacy_list_metadata(value) for value in obj]
    if not isinstance(obj, dict):
        return obj
    if obj.get("_type") == "List":
        if "feature" not in obj or set(obj) - {"_type", "feature", "length", "id"}:
            raise ValueError("Unsupported List feature metadata; refusing to guess its schema")
        feature = legacy_list_metadata(obj["feature"])
        length = obj.get("length", -1)
        if not isinstance(length, int) or isinstance(length, bool) or length < -1:
            raise ValueError("Invalid List length")
        if length == -1:
            return [feature]
        if isinstance(feature, dict) and not isinstance(feature.get("_type"), str):
            raise ValueError("Fixed-length List of structs requires a native List reader; do not reinterpret it as Sequence")
        return {"_type": "Sequence", "feature": feature, "length": length, "id": obj.get("id")}
    return {key: legacy_list_metadata(value) for key, value in obj.items()}


def enable_list_reader():
    import datasets
    import datasets.features.features as features
    if hasattr(datasets, "List"):
        return {"policy": "native_List", "datasets_version": datasets.__version__}
    if datasets.__version__ != "3.6.0":
        raise RuntimeError("The compatibility reader is tested for datasets==3.6.0; use the original pinned environment")
    original = features.generate_from_dict
    if not getattr(original, "_relu2_list_reader", False):
        def generate(obj):
            return original(legacy_list_metadata(obj))
        generate._relu2_list_reader = True
        features.generate_from_dict = generate
    # Exercise the exact deserialization route used by dataset/Parquet metadata.
    features.Features.from_dict({"choices": {"_type": "List", "feature": {"_type": "Value", "dtype": "string"}}})
    return {"policy": "serialized_List_to_legacy_list_v1", "datasets_version": datasets.__version__}
