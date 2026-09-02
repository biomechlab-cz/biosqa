"""Tests for the record-level data-quality report."""
import numpy as np

from biosqa.inference.data_quality import _rolling_std, record_quality


def _clean(fs=250.0, secs=20.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def test_clean_record_is_usable():
    q = record_quality(_clean(), 250.0)
    assert q.usable and q.completeness > 0.95 and not q.flags


def test_dropout_gap_detected():
    x = _clean(); x[1000:3000] = 0.0            # 8 s dropout at 250 Hz
    q = record_quality(x, 250.0)
    assert q.n_dropout_gaps >= 1 and q.longest_gap_s >= 7.0
    assert any("dropout" in f for f in q.flags)


def test_clipping_detected():
    x = _clean(); x = np.clip(x, -0.3, 0.3)     # saturate the rails
    q = record_quality(x, 250.0)
    assert q.clipping_frac > 0.05 and any("clipped" in f for f in q.flags)


def test_nan_missing_and_multichannel_worst_case():
    a = _clean(seed=1); b = _clean(seed=2); b[::2] = np.nan   # one broken lead
    q = record_quality(np.stack([a, b]), 250.0)
    assert q.missing_frac > 0.4 and not q.usable


def _eeg_uv(amp_uv=40.0, fs=250.0, secs=20.0, seed=3):
    """Alpha-band EEG in MICROVOLTS; multiply by 1e-6 for the volts an MNE loader actually returns."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    return amp_uv * (np.sin(2 * np.pi * 10.0 * t) + 0.3 * rng.standard_normal(len(t)))


def _dead_uv(n, dc_uv=0.0, dither_uv=1e-3, seed=7):
    """A DEAD channel as the hardware really hands one back: a stuck DC plus white dither. Never exactly
    constant -- which is precisely why a detector that only looks at |diff| against the channel's own
    typical |diff| cannot see it (the dither IS the channel's typical |diff|).

    Two physical regimes, and the difference between them is where this check has repeatedly broken:
      * ``dither_uv=1e-3`` -- about 1 ADC LSB. The QUIETEST a dead lead can plausibly be.
      * ``dither_uv=1.0``  -- the AMPLIFIER NOISE FLOOR (~1 uV RMS), which is what a floating /
        disconnected electrode actually dithers at. This is the realistic case, and it is ~1000x
        louder than 1 LSB -- an amplitude-only test tuned on the LSB fixture goes blind here.
    """
    rng = np.random.default_rng(seed)
    return dc_uv + dither_uv * rng.standard_normal(n)


def test_flatline_verdict_is_scale_invariant():
    """Regression: flatline used an ABSOLUTE tolerance, so the same trace in volts (what MNE hands back)
    scored 39% flatline while in microvolts it scored 0%. The unit must not change the verdict."""
    x = _eeg_uv()
    x[2000:3000] = 7.5                                  # 4 s genuinely dead: constant, off-rail, non-zero
    q_uv = record_quality(x, 250.0)
    q_v = record_quality(x * 1e-6, 250.0)               # identical trace, SI volts
    assert q_v.flatline_frac == q_uv.flatline_frac
    assert q_v.usable == q_uv.usable
    assert 0.15 < q_uv.flatline_frac < 0.25             # the dead segment only: 1000 of 5000 samples


def test_flatline_has_no_absolute_floor_left():
    """The relative tolerance still carried an absolute 1e-12 floor, so scale-invariance silently died
    below ~1e-9: this healthy EEG scaled by 1e-9 (an amplitude no floor may assume away) scored
    flatline_frac = 0.9998 -- a clean trace reported as a dead sensor. The ratio must be unit-free at
    EVERY amplitude, so all three scalings must agree exactly."""
    x = _eeg_uv()
    fracs = {record_quality(x * k, 250.0).flatline_frac for k in (1.0, 1e-6, 1e-9)}
    assert fracs == {0.0}


def test_low_voltage_eeg_in_volts_is_not_a_dead_sensor():
    """A healthy low-voltage EEG in volts used to score 79% flatline (2 uV) / 39% (5 uV) -> usable=False."""
    for amp_uv in (2.0, 5.0):
        q = record_quality(_eeg_uv(amp_uv=amp_uv) * 1e-6, 250.0)
        assert q.flatline_frac < 0.05 and q.usable
        assert not any("flatline" in f for f in q.flags)


def test_true_flat_channel_still_detected_at_any_scale():
    """The dead-sensor check itself must not regress: exactly-constant and all-zero stay flatline."""
    const = record_quality(np.full(5000, 3.5), 250.0)
    assert const.flatline_frac > 0.99 and not const.usable
    assert any("flatline" in f for f in const.flags)
    assert record_quality(np.full(5000, 3.5e-6), 250.0).flatline_frac == const.flatline_frac
    zeros = record_quality(np.zeros(5000), 250.0)
    assert zeros.flatline_frac > 0.99 and not zeros.usable


def test_dead_channel_with_adc_dither_is_flatline():
    """Audit repro: a channel that is dead END TO END but dithers by ~1 LSB (1e-9 V) read
    flatline_frac = 0.0008, flags = [] -> CLEAN. A dead sensor is never exactly constant in the real
    world, so the check has to survive dither. Detected by shape (the trace is white noise: no
    band-limited biosignal at any amplitude), which is scale-free -- so the verdict must not change
    when the same trace is scaled by 1e9."""
    dead = _dead_uv(5000, dither_uv=1e-3) * 1e-6        # ~1e-9 V of dither on a stuck lead
    q = record_quality(dead, 250.0)
    assert q.flatline_frac > 0.9 and not q.usable
    assert any("flatline" in f for f in q.flags)
    assert record_quality(dead * 1e9, 250.0).flatline_frac == q.flatline_frac


def test_half_dead_channel_is_flagged():
    """Audit repro: half live EEG, half dead-with-dither -- a half-dead sensor passing as PERFECT,
    the exact failure this check exists to prevent.

    The dither here is the AMPLIFIER NOISE FLOOR (1 uV RMS on a 30 uV channel), not 1 ADC LSB. That
    re-parameterisation is the whole point: this test previously used a ~1e-3 uV dither and so passed
    over a live bug. A local-excursion test gated on 0.1% of the channel's dynamic range only sees a
    dead stretch quieter than that; a real floating electrode dithers at ~3% of range, i.e. ~30x above
    the gate, and the detector reported flatline_frac = 0.000, completeness = 1.0, flags = [] -- CLEAN.
    A dead stretch is dead because it has no signal STRUCTURE, not merely because it is small, so the
    verdict must come from the same shape test that catches a fully dead lead."""
    x = _eeg_uv(amp_uv=30.0)
    x[2500:] = 7.5 + _dead_uv(2500, dither_uv=1.0)      # 10 s live alpha, then 10 s of a floating lead
    q = record_quality(x * 1e-6, 250.0)
    assert 0.4 < q.flatline_frac < 0.6                  # ~half the record is dead, and it says so
    assert any("flatline" in f for f in q.flags)


def test_partially_dead_channel_detected_across_noise_floor_and_extent():
    """The hole was a CLIFF, so pin the whole surface, not one point: a dead stretch must be found
    whatever fraction of the record it covers and wherever its noise floor sits between 1 LSB and the
    amplifier floor. The old amplitude-only rule fell off this cliff at ~0.1% of the channel's range
    (1 LSB: found; anything louder: flatline_frac 0.000, silently clean).

    The measured fraction runs ~1 s short of the truth: the shape window straddles the live/dead
    boundary there, so the first second of a dead stretch is not yet structureless. That is a
    resolution limit, not a miss -- it is bounded and reported, where the old behaviour was 0.000."""
    for dither_uv in (1e-3, 0.05, 0.3, 1.0):            # 1 LSB ... amplifier noise floor
        for dead_frac in (0.25, 0.5, 0.75):
            k = int(5000 * (1 - dead_frac))
            x = _eeg_uv(amp_uv=30.0)
            x[k:] = 7.5 + _dead_uv(5000 - k, dither_uv=dither_uv)
            q = record_quality(x * 1e-6, 250.0)
            assert dead_frac - 0.06 < q.flatline_frac <= dead_frac, (dither_uv, dead_frac, q.flatline_frac)
            assert not q.usable if dead_frac > 0.55 else True
            if dead_frac >= 0.5:                        # ... and it is loud enough to reach the user
                assert any("flatline" in f for f in q.flags), (dither_uv, dead_frac)


def _ecg_mv(fs=360.0, secs=30.0, seed=1):
    """Spiky ECG: QRS + T + baseline wander. Its 1-99 range is set by beats, not by its noise floor."""
    rng = np.random.default_rng(seed)
    n = int(fs * secs)
    t = np.arange(n) / fs
    x = 0.05 * np.sin(2 * np.pi * 0.25 * t)
    for k in range(int(secs) + 1):
        t0 = float(k)
        x += 1.0 * np.exp(-0.5 * ((t - t0) / 0.010) ** 2)
        x += -0.2 * np.exp(-0.5 * ((t - t0 - 0.03) / 0.010) ** 2)
        x += 0.2 * np.exp(-0.5 * ((t - t0 - 0.22) / 0.040) ** 2)
    return x + 0.03 * rng.standard_normal(n)


def _ppg(fs=64.0, secs=60.0, seed=2):
    """Smooth pulsatile PPG + respiration."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = (np.sin(2 * np.pi * 1.2 * t) + 0.35 * np.sin(2 * np.pi * 2.4 * t + 1.1)
         + 0.15 * np.sin(2 * np.pi * 0.25 * t))
    return x + 0.02 * rng.standard_normal(t.size)


def test_dead_stretch_detected_in_every_modality_at_the_noise_floor():
    """The detector is shared by all four modalities, so pin it on all of them -- the bug was found on
    EEG, but nothing about it was EEG-specific.

    The dead stretch is injected at the physically meaningful level: a fraction of what THIS channel's
    local activity is while it is ALIVE (that is what an amplifier noise floor is). Sweeping 2%..10%
    covers a real floating electrode. Calibrating instead against the 1-99 range would be wrong -- drift
    and artifact inflate that range, so 'a few percent of range' can be as loud as the live signal
    itself, which is not a dead sensor and must not be called one."""
    for name, x, fs in (("ecg", _ecg_mv(), 360.0), ("ppg", _ppg(), 64.0),
                        ("eeg", _eeg_uv(amp_uv=30.0) * 1e-6, 250.0)):
        clean = record_quality(x, fs)
        assert clean.flatline_frac == 0.0 and not clean.flags, name       # live trace stays clean
        k = x.size // 2
        live = x[:k]
        act = float(np.median(_rolling_std(live - np.median(live), int(round(0.5 * fs)))))
        for floor in (0.02, 0.05, 0.10):
            rng = np.random.default_rng(0)
            y = x.copy()
            y[k:] = float(np.median(x)) + floor * act * rng.standard_normal(x.size - k)
            q = record_quality(y, fs)
            assert 0.4 < q.flatline_frac < 0.6, (name, floor, q.flatline_frac)
            assert any("flatline" in f for f in q.flags), (name, floor)


def test_resting_slow_signal_is_not_a_dead_sensor():
    """The false-flag trap on the other side, and the reason the shape test is gated on amplitude.

    A live EDA sensor at rest (no SCR, slow tonic drift, quantised to the recorder's 1 mS step) is,
    inside any short window, a CONSTANT PLUS WHITE NOISE -- locally indistinguishable from a dead lead.
    A per-window whiteness test applied on its own therefore calls a resting EDA channel a dead sensor
    (measured: real 8 Hz UT-Dallas EDA jumped from flatline_frac 0.10 to 0.46). It is only safe because
    the stretch must ALSO be still relative to what THIS channel does when it is alive; a resting EDA
    is quiet everywhere, so its quiet stretches are not anomalous FOR IT and it stays clean."""
    fs, secs = 8.0, 300.0
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * secs)) / fs
    eda = 4.0 + 0.3 * np.sin(2 * np.pi * 0.005 * t)          # microsiemens: slow tonic drift
    for t0 in (60.0, 150.0, 240.0):                          # a few skin-conductance responses
        d = np.clip(t - t0, 0.0, None)
        eda += 0.8 * (np.exp(-d / 6.0) - np.exp(-d / 0.8)) * (t >= t0)
    eda = np.round(eda + 0.002 * rng.standard_normal(t.size), 3)   # 1 mS quantisation, as an E4 gives
    q = record_quality(eda, fs)
    assert q.flatline_frac < 0.05 and q.usable
    assert not any("flatline" in f for f in q.flags)


def test_broadband_artifact_burst_is_not_a_dead_sensor():
    """The other thing a whiteness test must not confuse: EMG/motion artifact is broadband too. A loud
    white BURST on a live lead is the opposite of a dead sensor, so 'structureless' may only ever mean
    'dead' when the stretch is also STILL -- a stretch moving through a real part of the channel's
    dynamic range is never a dead sensor, however white it looks."""
    rng = np.random.default_rng(0)
    x = _eeg_uv(amp_uv=30.0)
    x[1000:2000] += 60.0 * rng.standard_normal(1000)         # 4 s of loud broadband artifact
    q = record_quality(x * 1e-6, 250.0)
    assert q.flatline_frac == 0.0
    assert not any("flatline" in f for f in q.flags)


def test_oversampled_slow_signal_is_not_flat():
    """The trap the amplitude-ratio design must survive: a genuine 0.1 Hz EDA trace sampled at 1 kHz.
    Its per-sample increments are ~1e-4 of its amplitude (asserted below), so ANY per-sample-difference
    test -- absolute or relative to the signal's own amplitude -- calls it a dead sensor. Judged over a
    half-second window instead, it plainly moves: a real slow signal travels, a dead one does not."""
    fs = 1000.0
    t = np.arange(int(60 * fs)) / fs
    eda = 5.0 + 2.0 * np.sin(2 * np.pi * 0.1 * t)       # microsiemens: tonic level + a slow wave
    assert np.median(np.abs(np.diff(eda))) < 1e-3 * np.ptp(eda)   # the trap: ~2e-4 of its own range
    q = record_quality(eda, fs)
    assert q.flatline_frac == 0.0 and q.usable
    assert not any("flatline" in f for f in q.flags)


def test_dead_lead_among_live_leads_is_caught():
    """The realistic dead-electrode case through the real [C, L] path the worker uses: one disconnected
    lead in an otherwise healthy montage. Worst-case channel reduction must surface it -- it read
    flatline_frac = 0.0006, usable = True before."""
    live = _eeg_uv(seed=4) * 1e-6
    montage = np.stack([live, _eeg_uv(seed=5) * 1e-6, _dead_uv(live.size, dc_uv=1e-3) * 1e-6])
    q = record_quality(montage, 250.0)
    assert q.flatline_frac > 0.9 and not q.usable
    assert any("flatline" in f for f in q.flags)
    assert record_quality(montage[:2], 250.0).usable    # ... and the two live leads alone stay clean


def test_completeness_counts_all_dropouts_not_just_longest():
    """Regression: completeness must reflect TOTAL dropout duration, not only the longest gap."""
    import numpy as np
    from biosqa.inference.data_quality import record_quality
    fs = 250
    n = fs * 60
    x = np.sin(np.arange(n) / fs).astype(float)
    for k in range(20):                       # 20 separate 1 s zero-dropouts = 20 s of 60 s lost
        s = k * 3 * fs
        x[s:s + fs] = 0.0
    rq = record_quality(x, fs)
    assert rq.n_dropout_gaps >= 15
    assert rq.completeness < 0.75             # ~0.67, not the old ~0.98 longest-gap value
