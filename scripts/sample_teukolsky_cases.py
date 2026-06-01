from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

import numpy as np


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7.0 * SECONDS_PER_DAY
GMSUN_SEC = 4.92549095e-6


def cadence_from_tmax(tmax: float) -> float:
    if tmax > SECONDS_PER_WEEK:
        return SECONDS_PER_WEEK
    if tmax > SECONDS_PER_DAY:
        return SECONDS_PER_DAY
    return SECONDS_PER_HOUR


def format_value(value: float) -> str:
    return f"{value:.10e}"


def parse_numeric_row(line: str, columns: list[str]) -> dict[str, float]:
    values = line.strip().split()
    if len(values) != len(columns):
        raise ValueError(f"expected {len(columns)} columns, got {len(values)} in line: {line[:200]!r}")
    return {name: float(value) for name, value in zip(columns, values)}


def read_metadata(path: Path) -> tuple[list[str], list[str]]:
    comments: list[str] = []
    columns: list[str] | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue
            columns = line.strip().split()
            break
    if columns is None:
        raise ValueError(f"missing header row in {path}")
    return comments, columns


def read_last_time(path: Path) -> float:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        block = 16384
        pos = max(0, end - block)
        fh.seek(pos)
        tail = fh.read().decode("utf-8", errors="ignore").strip().splitlines()
    for line in reversed(tail):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("t "):
            return float(stripped.split()[0])
    raise ValueError(f"failed to locate last data row in {path}")


def sample_times(tmax: float) -> np.ndarray:
    dt = cadence_from_tmax(tmax)
    grid = np.arange(0.0, tmax + 0.5 * dt, dt, dtype=float)
    grid = grid[grid <= tmax]
    if grid.size == 0 or abs(grid[-1] - tmax) > 1e-9:
        grid = np.append(grid, tmax)
    return grid


def interpolate_row(prev_row: dict[str, float], next_row: dict[str, float], target_t: float, columns: list[str]) -> dict[str, float]:
    t0 = prev_row["t"]
    t1 = next_row["t"]
    if abs(t1 - t0) < 1e-15:
        out = dict(next_row)
        out["t"] = target_t
        return out
    alpha = (target_t - t0) / (t1 - t0)
    out = {}
    for column in columns:
        if column == "t":
            out[column] = target_t
        else:
            out[column] = prev_row[column] + alpha * (next_row[column] - prev_row[column])
    return out


def stream_sample_rows(path: Path, columns: list[str], targets: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    target_index = 0
    prev_row: dict[str, float] | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("t "):
                continue
            row = parse_numeric_row(stripped, columns)
            if prev_row is None:
                prev_row = row
                while target_index < len(targets) and targets[target_index] <= row["t"]:
                    first = dict(row)
                    first["t"] = float(targets[target_index])
                    rows.append(first)
                    target_index += 1
                continue
            while target_index < len(targets) and targets[target_index] <= row["t"]:
                target_t = float(targets[target_index])
                rows.append(interpolate_row(prev_row, row, target_t, columns))
                target_index += 1
            prev_row = row
            if target_index >= len(targets):
                break
    if prev_row is None:
        raise ValueError(f"no data rows found in {path}")
    while target_index < len(targets):
        last = dict(prev_row)
        last["t"] = float(targets[target_index])
        rows.append(last)
        target_index += 1
    return rows


def finite_difference_jacobian(p: float, e: float) -> np.ndarray:
    from teukolsky import KerrGeoOrbit

    dp = max(1e-6, 1e-5 * abs(p))
    de = max(1e-8, 1e-5 * max(abs(e), 1e-3))

    def orbit_quantities(pp: float, ee: float) -> tuple[float, float]:
        orb = KerrGeoOrbit(0.0, float(pp), float(ee), 1.0)
        return float(orb.energy), float(orb.angular_momentum)

    e_minus = max(0.0, e - de)
    e_plus = min(0.95, e + de)

    e_p_plus, l_p_plus = orbit_quantities(p + dp, e)
    e_p_minus, l_p_minus = orbit_quantities(p - dp, e)
    dE_dp = (e_p_plus - e_p_minus) / (2.0 * dp)
    dL_dp = (l_p_plus - l_p_minus) / (2.0 * dp)

    if e_plus > e_minus:
        e_e_plus, l_e_plus = orbit_quantities(p, e_plus)
        e_e_minus, l_e_minus = orbit_quantities(p, e_minus)
        dE_de = (e_e_plus - e_e_minus) / (e_plus - e_minus)
        dL_de = (l_e_plus - l_e_minus) / (e_plus - e_minus)
    else:
        dE_de = 0.0
        dL_de = 0.0

    return np.array([[dE_dp, dE_de], [dL_dp, dL_de]], dtype=float)


def total_fluxes(
    p: float,
    e: float,
    lmax: int,
    nmax: int,
    device_id: int,
    accelerator_resolution: int | None,
) -> tuple[float, float]:
    from teukolsky import KerrGeoOrbit
    from teukolsky.modes import solve_point_particle_mode

    orbit = KerrGeoOrbit(0.0, float(p), float(e), 1.0)
    energy_flux = 0.0
    angular_flux = 0.0
    for ell in range(2, lmax + 1):
        for m in range(1, ell + 1):
            for n in range(-nmax, nmax + 1):
                mode = solve_point_particle_mode(
                    -2,
                    ell,
                    m,
                    orbit,
                    n=n,
                    k=0,
                    accelerator="gpu",
                    device_id=device_id,
                    accelerator_resolution=accelerator_resolution,
                )
                # This script only sums positive-m modes. For the
                # Schwarzschild equatorial orbits used here (a=0, x=1),
                # the negative-m contribution is equal, so the total flux
                # is twice the positive-m sum.
                energy_flux += 2.0 * float(mode.fluxes.energy.real)
                angular_flux += 2.0 * float(mode.fluxes.angular_momentum.real)
    return energy_flux, angular_flux


def compute_row(
    row: dict[str, float],
    device_id: int,
    lmax: int,
    nmax: int,
    accelerator_resolution: int | None,
) -> dict[str, float]:
    from teukolsky import KerrGeoOrbit

    p = float(row["p"])
    e = float(row["e"])
    M = float(row["M"])
    mu = float(row["mu"])
    M_sec = M * GMSUN_SEC
    scale = (mu / M) / M_sec

    orbit = KerrGeoOrbit(0.0, p, e, 1.0)
    energy_flux_raw, angular_flux_raw = total_fluxes(
        p,
        e,
        lmax=lmax,
        nmax=nmax,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )
    Edot = -energy_flux_raw * scale
    Lzdot = -angular_flux_raw * scale

    jacobian = finite_difference_jacobian(p, e)
    rhs = np.array([Edot, Lzdot], dtype=float)
    try:
        pdot, edot = np.linalg.solve(jacobian, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"singular Jacobian at p={p}, e={e}") from exc

    out = dict(row)
    out["pdot"] = float(pdot)
    out["edot"] = float(edot)
    out["E"] = float(orbit.energy)
    out["Lz"] = float(orbit.angular_momentum)
    out["Edot"] = float(Edot)
    out["Lzdot"] = float(Lzdot)
    out["Omega_r"] = float(orbit.omega_r / M_sec)
    out["Omega_phi"] = float(orbit.omega_phi / M_sec)
    return out


def write_output(path: Path, comments: list[str], columns: list[str], rows: Iterable[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for comment in comments:
            fh.write(comment + "\n")
        fh.write("# teukolsky_recomputed=1\n")
        fh.write(" ".join(columns) + "\n")
        for row in rows:
            fh.write(" ".join(format_value(float(row[column])) for column in columns) + "\n")


def process_case(case_dir: Path, device_id: int, lmax: int, nmax: int, accelerator_resolution: int | None) -> None:
    src = case_dir / "PN5.txt"
    dst = case_dir / "teukolsky.txt"
    comments, columns = read_metadata(src)
    tmax = read_last_time(src)
    targets = sample_times(tmax)
    sampled_rows = stream_sample_rows(src, columns, targets)
    computed = [
        compute_row(
            row,
            device_id=device_id,
            lmax=lmax,
            nmax=nmax,
            accelerator_resolution=accelerator_resolution,
        )
        for row in sampled_rows
    ]
    write_output(dst, comments, columns, computed)
    print(
        f"[gpu {device_id}] wrote {dst} with {len(computed)} rows "
        f"(l=2..{lmax}, n={-nmax}..{nmax}, resolution={accelerator_resolution})",
        flush=True,
    )


def worker(device_id: int, cases: list[Path], lmax: int, nmax: int, accelerator_resolution: int | None) -> None:
    os.environ["HIP_VISIBLE_DEVICES"] = str(device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    for case_dir in cases:
        process_case(case_dir, device_id=0, lmax=lmax, nmax=nmax, accelerator_resolution=accelerator_resolution)


def chunk_cases(cases: list[Path], ngpu: int) -> list[list[Path]]:
    groups: list[list[Path]] = [[] for _ in range(ngpu)]
    for idx, case_dir in enumerate(cases):
        groups[idx % ngpu].append(case_dir)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample PN5 case files and recompute sparse Teukolsky tracks.")
    parser.add_argument("--base-dir", type=Path, default=Path("/public/home/licm/符号回归"))
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case directory names such as case1 case2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--nmax", type=int, default=3)
    parser.add_argument("--accelerator-resolution", type=int, default=8193)
    args = parser.parse_args()

    all_cases = sorted(path for path in args.base_dir.glob("case*") if path.is_dir())
    if args.cases:
        wanted = set(args.cases)
        all_cases = [path for path in all_cases if path.name in wanted]
    if not all_cases:
        raise SystemExit("no case directories selected")

    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    groups = chunk_cases(all_cases, len(gpu_ids))
    processes: list[mp.Process] = []
    for gpu_id, case_group in zip(gpu_ids, groups):
        if not case_group:
            continue
        proc = mp.Process(
            target=worker,
            args=(gpu_id, case_group, args.lmax, args.nmax, args.accelerator_resolution),
            daemon=False,
        )
        proc.start()
        processes.append(proc)
    exit_code = 0
    for proc in processes:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
