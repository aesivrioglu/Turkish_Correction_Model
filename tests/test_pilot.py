from trgec.pilot import halve_batch, is_out_of_memory


def test_is_out_of_memory_reads_torch_message():
    log = "adım 1\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.\n"
    assert is_out_of_memory(log)
    assert not is_out_of_memory("ValueError: bilinmeyen model\n")
    assert not is_out_of_memory("")


def test_halve_batch_keeps_effective_batch_size():
    assert halve_batch(16, 1) == (8, 2)
    assert halve_batch(8, 2) == (4, 4)
    assert halve_batch(4, 4) == (2, 8)
    assert halve_batch(2, 8) == (1, 16)
    assert halve_batch(1, 16) is None
    # Zincirin her halkasında etkin grup boyutu 16 kalır.
    setting = (16, 1)
    while setting is not None:
        assert setting[0] * setting[1] == 16
        setting = halve_batch(*setting)
