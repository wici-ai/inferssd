import gguf


def test_dspark_tensor_mapping() -> None:
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.DFLASH, 5)
    expected = {
        "model.fc.weight": "fc.weight",
        "model.markov_head.markov_w1.weight": "markov_w1.weight",
        "model.markov_head.markov_w2.weight": "markov_w2.weight",
        "model.confidence_head.proj.weight": "conf_proj.weight",
        "model.confidence_head.proj.bias": "conf_proj.bias",
        "model.layers.4.self_attn.q_proj.weight": "blk.4.attn_q.weight",
    }

    for source_name, gguf_name in expected.items():
        assert tensor_map.get_name(
            source_name,
            try_suffixes=(".weight", ".bias"),
        ) == gguf_name


def test_dspark_heads_are_valid_dflash_tensors() -> None:
    tensors = set(gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DFLASH])
    assert {
        gguf.MODEL_TENSOR.FC,
        gguf.MODEL_TENSOR.ENC_OUTPUT_NORM,
        gguf.MODEL_TENSOR.DSPARK_MARKOV_W1,
        gguf.MODEL_TENSOR.DSPARK_MARKOV_W2,
        gguf.MODEL_TENSOR.DSPARK_CONF_PROJ,
    } <= tensors
