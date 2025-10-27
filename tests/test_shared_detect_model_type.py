from scripts._shared import detect_model_type


def test_detect_model_type_v1():
    sd = {"blocks.0.attn.qkv.weight": object(), "blocks.0.attn.attn_mask": object()}
    assert detect_model_type(sd) == "tiny_gpt"


def test_detect_model_type_v2():
    sd = {"blocks.0.attn.key.weight": object(), "blocks.0.attn.mask": object()}
    assert detect_model_type(sd) == "tiny_gpt_v2"

