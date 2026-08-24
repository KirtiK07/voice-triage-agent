from voice_agent.audio_utils import resample_int16


def test_same_rate_is_a_no_op():
    audio = b"\x01\x02\x03\x04\x05\x06"
    assert resample_int16(audio, 16000, 16000) is audio


def test_output_length_scales_with_rate_ratio():
    one_second_at_22050 = b"\x00\x00" * 22050
    result = resample_int16(one_second_at_22050, 22050, 16000)
    n_samples_out = len(result) // 2
    assert abs(n_samples_out - 16000) <= 1  # allow for integer-truncation rounding


def test_silence_stays_silence():
    silence = b"\x00\x00" * 1000
    result = resample_int16(silence, 22050, 16000)
    assert result == b"\x00\x00" * (len(result) // 2)
