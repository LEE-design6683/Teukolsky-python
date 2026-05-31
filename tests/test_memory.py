import gc
import tracemalloc

from teukolsky import KerrGeoOrbit, TeukolskyPointParticleMode


def test_repeated_point_particle_mode_does_not_accumulate_memory() -> None:
    orbit = KerrGeoOrbit(0.3, 10.0, 0.0, 1.0)

    tracemalloc.start()
    try:
        # Warm-up run to populate any one-time caches.
        TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        for _ in range(3):
            TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)
        gc.collect()
        after_first_batch = tracemalloc.get_traced_memory()[0]

        for _ in range(3):
            TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)
        gc.collect()
        after_second_batch = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()

    first_growth = after_first_batch - baseline
    second_growth = after_second_batch - after_first_batch

    assert first_growth < 8_000_000, f"unexpected first-batch growth: {first_growth} bytes"
    assert second_growth < 512_000, f"possible repeated-call leak: {second_growth} bytes"
