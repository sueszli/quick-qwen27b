import qwen


def test_vram_budget_fits_a_24gb_card():
    weights_gib = 18.8 + 0.9
    kv_gib = qwen.CTX * 16 * 2 * 4 * 256 * 1 / 2**30
    assert weights_gib + kv_gib + 1.5 < 24


def test_sampling_matches_model_card():
    assert qwen.THINKING == {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
    assert qwen.INSTRUCT["temperature"] == 0.7
