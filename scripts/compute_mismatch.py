#!/usr/bin/env python3
"""EMRI waveform mismatch: Teukolsky mode-sum vs FEW (reference).

GPU-accelerated waveform generation.  Computes overlap / mismatch
for Schwarzschild and Kerr eccentric-equatorial inspirals.

Key references:
- McCart+2021 (2109.00056): NK-Teukolsky mismatch < 1e-4
- Khalvati+2024 (2410.17310): PN5 vs Teukolsky flux ~few % for Kerr eq
- FEW validation: adiabatic Schwarzschild mismatch < 1e-7 (same mode set)
"""

from __future__ import annotations

import json, time, sys
from pathlib import Path

import numpy as np

# --- mismatch ---

def overlap(h1: np.ndarray, h2: np.ndarray) -> float:
    num = abs(np.vdot(h2, h1))
    den = np.sqrt(np.vdot(h1, h1).real * np.vdot(h2, h2).real)
    return float(num / den) if den > 1e-300 else 0.0

def mismatch(h1, h2):
    return 1.0 - overlap(h1, h2)

# --- test cases (E M R I parameter space) ---
# Literature benchmarks: arXiv:2410.17310, arXiv:2109.00056

CASES = [
    # Schwarzschild eccentric — matched-mode test
    {"label": "Schw p=10 e=0.2  lmax=2 nmax=1",     "M": 1e6, "a": 0.0, "p0": 10.0, "e0": 0.2,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 1e-6},
    {"label": "Schw p=12 e=0.4  lmax=2 nmax=1",     "M": 1e6, "a": 0.0, "p0": 12.0, "e0": 0.4,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 1e-6},
    {"label": "Schw p=15 e=0.1  lmax=3 nmax=2",     "M": 1e6, "a": 0.0, "p0": 15.0, "e0": 0.1,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 3, "nmax": 2, "mm_thr": 1e-5},
    # Kerr equatorial eccentric
    {"label": "Kerr a=0.5 p=10 e=0.2  lmax=2 nmax=1","M": 1e6, "a": 0.5, "p0": 10.0, "e0": 0.2,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 5e-3},
    {"label": "Kerr a=0.9 p=10 e=0.2  lmax=2 nmax=1","M": 1e6, "a": 0.9, "p0": 10.0, "e0": 0.2,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 5e-3},
    {"label": "Kerr a=0.7 p=12 e=0.3  lmax=2 nmax=1","M": 1e6, "a": 0.7, "p0": 12.0, "e0": 0.3,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 5e-3},
    # Higher mode resolution
    {"label": "Kerr a=0.5 p=10 e=0.2  lmax=3 nmax=2","M": 1e6, "a": 0.5, "p0": 10.0, "e0": 0.2,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 3, "nmax": 2, "mm_thr": 5e-3},
    # EMRI-like (small mass ratio)
    {"label": "EMRI mu=1  a=0.5 p=10 e=0.2",         "M": 1e6, "a": 0.5, "p0": 10.0, "e0": 0.2,
     "x0": 1.0, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 5e-3,
     "mu": 1.0},
    # Generic Kerr non-equatorial: Teukolsky mode-sum vs FEW Pn5AAK
    # Both use the same PN5 trajectory; the mismatch measures AAK model error
    {"label": "Pn5AAK a=0.5 p=10 e=0.2 x=0.7",       "M": 1e6, "a": 0.5, "p0": 10.0, "e0": 0.2,
     "x0": 0.7, "T_yr": 1e-6, "dt": 10.0, "ell": 2, "nmax": 1, "mm_thr": 0.10,
     "ref": "pn5aak"},
]


def mode_list(ell_max, n_max):
    out = []
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m == 0: continue
            for n in range(-n_max, n_max + 1):
                out.append((ell, m, n, 0))
    return out


def run(accelerator="gpu"):
    from teukolsky import (
        generate_schwarzschild_eccentric_adiabatic_waveform,
        generate_equatorial_eccentric_adiabatic_waveform,
        generate_generic_eccentric_adiabatic_waveform,
        source_frame_radius,
    )

    results = {}
    t_total = time.time()

    for case in CASES:
        label = case["label"]
        print(f"\n{'='*70}\n  {label}\n{'='*70}")

        M_val = case["M"]
        mu_val = case.get("mu", 10.0)
        a_val = case["a"]; p0 = case["p0"]; e0 = case["e0"]; x0 = case["x0"]
        T_s = case["T_yr"] * 365.25 * 86400
        dt = case["dt"]
        time_arr = np.arange(0.0, T_s + 0.5*dt, dt, dtype=float)
        time_arr = time_arr[time_arr <= T_s]
        ell_max, n_max = case["ell"], case["nmax"]
        mm_teuk = mode_list(ell_max, n_max)
        mm_few = [(l, m, k, n) for l, m, n, k in mm_teuk]

        radius = source_frame_radius(1.0, mu_val)
        common = dict(
            M=M_val, mu=mu_val, time=time_arr,
            theta=1.0, phi=0.3, radius=radius,
            trajectory_dt=10.0, mode_indices=mm_teuk,
            trajectory_ell_max=ell_max, trajectory_n_max=n_max,
            accelerator=accelerator, device_id=0,
        )

        # --- Teukolsky waveform ---
        t0 = time.time()
        try:
            if ref_type == "pn5aak":
                wf = generate_generic_eccentric_adiabatic_waveform(
                    M=M_val, mu=mu_val, a=a_val, p0=p0, e0=e0, x0=x0,
                    time=time_arr,
                    theta=1.0, phi=0.3, radius=radius,
                    trajectory_dt=10.0, mode_indices=mm_teuk,
                    trajectory_ell_max=ell_max, trajectory_n_max=1, trajectory_k_max=1,
                    waveform_ell_max=ell_max, waveform_n_max=1, waveform_k_max=1,
                    accelerator=accelerator, device_id=0,
                )
            elif a_val == 0.0:
                wf = generate_schwarzschild_eccentric_adiabatic_waveform(
                    p0=p0, e0=e0, **common,
                )
            else:
                wf = generate_equatorial_eccentric_adiabatic_waveform(
                    a=a_val, p0=p0, e0=e0, x=x0, **common,
                )
            h_our = wf.waveform.complex_strain
            t_our = time.time() - t0
            print(f"  Teukolsky: {t_our:.1f}s  samples={len(h_our)}  modes={len(mm_teuk)}")
        except Exception as e:
            print(f"  Teukolsky FAILED: {e}")
            results[label] = {"status": "teukolsky_error", "error": str(e)}
            continue

        # --- FEW reference ---
        ref_type = case.get("ref", "few")

        if ref_type == "pn5aak":
            try:
                from few.waveform import Pn5AAKWaveform
                t0 = time.time()
                few_wf = Pn5AAKWaveform(force_backend="cpu")
                h_few = np.asarray(
                    few_wf(M_val, mu_val, a_val, p0, e0, x0,
                           dist=1.0, qS=1.0, phiS=0.3, qK=1.0, phiK=0.3,
                           T=case["T_yr"], dt=dt, mich=False),
                    dtype=np.complex128,
                )
                t_few = time.time() - t0
            except Exception as e:
                print(f"  Pn5AAK unavailable: {e}")
                results[label] = {"status": "no_few", "t_our_s": t_our}
                continue

        elif a_val == 0.0:
            try:
                from few.waveform import FastSchwarzschildEccentricFlux
                few_wf = FastSchwarzschildEccentricFlux(force_backend="cpu")
                h_few = np.asarray(
                    few_wf(M_val, mu_val, p0, e0, common["theta"], common["phi"],
                           T=case["T_yr"], dt=dt, dist=1.0,
                           mode_selection=mm_few, include_minus_mkn=False),
                    dtype=np.complex128,
                )
            else:
                from few.waveform import FastKerrEccentricEquatorialFlux
                few_wf = FastKerrEccentricEquatorialFlux(
                    force_backend="cpu", frame="source", return_list=False,
                    lmax=ell_max, nmax=n_max,
                )
                h_few = np.asarray(
                    few_wf(M_val, mu_val, a_val, p0, e0, x0,
                           common["theta"], common["phi"],
                           T=case["T_yr"], dt=dt, dist=1.0,
                           mode_selection=mm_few, include_minus_mkn=False),
                    dtype=np.complex128,
                )
            t_few = time.time() - t0
            print(f"  FEW:        {t_few:.1f}s  samples={len(h_few)}")
        except Exception as e:
            print(f"  FEW unavailable: {e}")
            results[label] = {"status": "no_few", "t_our_s": t_our}
            continue

        # Align lengths
        n_min = min(len(h_our), len(h_few))
        h_our, h_few = h_our[:n_min], h_few[:n_min]

        ov = overlap(h_our, h_few)
        mm = 1.0 - ov
        amp_r = float(np.mean(np.abs(h_our) / np.maximum(np.abs(h_few), 1e-300)))
        max_re = float(np.max(np.abs(h_our - h_few) / np.maximum(np.abs(h_few), 1e-300)))
        passed = mm < case["mm_thr"]
        status = "PASS" if passed else "FAIL"

        print(f"  overlap = {ov:.12f}  mismatch = {mm:.3e}  amp_ratio = {amp_r:.4f}  max_rel_err = {max_re:.3e}")
        print(f"  threshold = {case['mm_thr']:.1e}  →  {status}")

        results[label] = {
            "status": status,
            "overlap": float(ov), "mismatch": float(mm),
            "amp_ratio": float(amp_r), "max_rel_err": float(max_re),
            "t_our_s": float(t_our), "t_few_s": float(t_few),
            "n_samples": n_min, "n_modes": len(mm_teuk),
            "ell_max": ell_max, "n_max": n_max,
            "a": a_val, "p0": p0, "e0": e0,
        }

    t_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  Total: {t_total:.0f}s")
    print(f"{'='*70}")

    # Summary
    print(f"\n{'Case':<45s} {'Overlap':>14s} {'Mismatch':>10s} {'Status':>8s}")
    print("-" * 77)
    passed_n = 0; failed_n = 0
    for label, r in results.items():
        if r["status"] in ("teukolsky_error", "no_few"):
            print(f"  {label:<43s} {'—':>14s} {'—':>10s} {r['status']:>8s}")
        else:
            ov_s = f"{r['overlap']:.10f}" if r['overlap'] >= 0.9999 else f"{r['overlap']:.6f}"
            print(f"  {label:<43s} {ov_s:>14s} {r['mismatch']:>10.2e} {r['status']:>8s}")
            if r["status"] == "PASS": passed_n += 1
            else: failed_n += 1

    print(f"\n  {passed_n} passed, {failed_n} failed")

    Path("/public/home/licm/Teukolsky/测试/mismatch_results.json").parent.mkdir(parents=True, exist_ok=True)
    with open("/public/home/licm/Teukolsky/测试/mismatch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → /public/home/licm/Teukolsky/测试/mismatch_results.json")

    return results


if __name__ == "__main__":
    acc = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    run(acc)
