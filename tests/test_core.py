import numpy as np

from flyover_scrub.attack import score_candidates, whitened_spectrogram
from flyover_scrub.physics import (
    SPEED_OF_SOUND,
    emission_geometry,
    make_track,
    received_doppler,
    synthesize_flyover,
)
from flyover_scrub.scrub import ScrubConfig, scrub

SR = 8000


def _straight(v=80.0):
    return make_track(0.0, 200.0, start_xy=(-8000.0, 500.0), altitude=600.0, speed=v, heading_deg=0.0)


def test_doppler_limits_match_textbook():
    trk = _straight(80.0)
    _, g, _ = emission_geometry(trk, np.zeros(3))
    # far approach / far recede -> c/(c -/+ v) for a source moving straight at you
    assert abs(g[5] - SPEED_OF_SOUND / (SPEED_OF_SOUND - 80.0)) < 0.01
    assert abs(g[-5] - SPEED_OF_SOUND / (SPEED_OF_SOUND + 80.0)) < 0.01


def test_emission_time_is_not_just_a_delay():
    """The naive model (velocity at reception time) gives a differently shaped curve."""
    trk = _straight()
    obs = np.zeros(3)
    t_r = np.linspace(80, 130, 200)
    g_right, _, _ = received_doppler(trk, obs, t_r)
    _, g_e, _ = emission_geometry(trk, obs)
    g_naive = np.interp(t_r, trk.t, g_e)
    best = min(np.abs(np.interp(t_r + s, t_r, g_naive) - g_right).max() for s in np.linspace(-10, 10, 201))
    assert best > 0.01  # no time shift reconciles them


def test_synth_puts_energy_on_predicted_curve():
    trk = _straight()
    obs = np.zeros(3)
    n = SR * 60
    x = synthesize_flyover(trk, obs, 70.0, n, SR, f0=100.0, broadband=0.0)
    spec = whitened_spectrogram(x + 1e-3 * np.random.default_rng(0).normal(size=n), SR)
    g, _, _ = received_doppler(trk, obs, 70.0 + spec.times)
    bins = np.rint(100.0 * g / spec.df).astype(int)
    on = spec.S[bins, np.arange(len(bins))].mean()
    off = spec.S[bins + 15, np.arange(len(bins))].mean()
    assert on > off + 2.0


def test_attack_prefers_truth_and_scrub_removes_it():
    rng = np.random.default_rng(3)
    trk = make_track(0.0, 260.0, start_xy=(-9000.0, -400.0), altitude=700.0, speed=70.0, heading_deg=5.0,
                     turn_rate_deg_s=-1.0, turn_start=100.0, turn_duration=25.0)
    obs = np.zeros(3)
    n = SR * 80
    plane = synthesize_flyover(trk, obs, 100.0, n, SR, f0=90.0, rng=rng)
    bg = rng.normal(size=n)
    plane *= 10 ** (-28 / 20) * bg.std() / plane[plane != 0].std()
    mix = bg + plane
    cands = np.array([[0, 0, 0], [0, 800, 0], [600, -600, 0], [-1500, 900, 0]], float)
    kw = dict(f0_range=(87.0, 93.0), offsets=np.arange(-2.0, 2.01, 0.25))

    res = score_candidates(whitened_spectrogram(mix, SR), [trk], 100.0, cands, **kw)
    assert np.argmax(res.scores) == 0

    y, _ = scrub(mix, SR, ScrubConfig(seed=0))
    res2 = score_candidates(whitened_spectrogram(y, SR), [trk], 100.0, cands, **kw)
    ctrl = score_candidates(whitened_spectrogram(bg, SR), [trk], 100.0, cands, **kw)
    # After scrubbing, the true location must score like it does with no aircraft
    # at all. (Comparing argmax over 4 candidates is meaningless: geometry alone
    # can make truth #1 even in the control.)
    assert res.scores[0] - ctrl.scores[0] > 0.3
    assert res2.scores[0] - ctrl.scores[0] < 0.03


def test_scrub_preserves_foreground_and_level():
    rng = np.random.default_rng(1)
    n = SR * 20
    t = np.arange(n) / SR
    bg = 0.01 * rng.normal(size=n)
    # bursty "speech-like" harmonic content: short syllables, fast pitch movement
    env = (np.sin(2 * np.pi * 2.5 * t) > 0.3).astype(float)
    f0 = 180 + 40 * np.sin(2 * np.pi * 3.0 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    voice = env * sum(0.2 / k * np.sin(k * ph) for k in range(1, 8))
    y, rep = scrub(bg + voice, SR, ScrubConfig(seed=0))
    assert rep.ridges_found <= 3  # syllables must not be chained into long ridges
    on = env > 0
    err = np.std(y[on] - (bg + voice)[on]) / np.std(voice[on])
    assert err < 0.35
    assert abs(20 * np.log10(np.std(y) / np.std(bg + voice))) < 1.0
