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


def test_training_status_separates_healthy_runs_from_broken_ones():
    from trgec.pilot import STATUS_FAILED, STATUS_NAN, STATUS_NO_LEARNING, STATUS_OK, training_status

    assert training_status(True, 0.58, 0.20) == STATUS_OK
    # Betik hatasız bitti ama kayıp hiç düşmedi: model öğrenmedi.
    assert training_status(True, 11.93, 11.93) == STATUS_NO_LEARNING
    # Kayıp NaN oldu: adım numarası kayıtlıysa da, yalnızca NaN değer varsa da yakalanır.
    assert training_status(False, float("nan"), float("nan"), nan_step=20) == STATUS_NAN
    assert training_status(True, 0.5, float("nan")) == STATUS_NAN
    # Özet yok (ör. bellek hiçbir grup boyutunda yetmedi).
    assert training_status(False, None, None) == STATUS_FAILED
