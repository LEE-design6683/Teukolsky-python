from __future__ import annotations

from teukolsky.geodesics import circular_orbit, equatorial_eccentric_orbit, generic_orbit, spherical_orbit
from teukolsky.modes import solve_point_particle_mode
from teukolsky.radial import solve_radial


def KerrGeoOrbit(a: float, p: float, e: float, x: float):
    if e == 0.0 and abs(x) == 1.0:
        return circular_orbit(a, p)
    if e == 0.0:
        return spherical_orbit(a, p, x)
    if abs(x) == 1.0:
        return equatorial_eccentric_orbit(a, p, e, x)
    return generic_orbit(a, p, e, x)


def _parse_radial_method(method, domain):
    if isinstance(method, (tuple, list)):
        if not method:
            raise ValueError("method tuple/list must not be empty")
        method_name = method[0]
        method_domain = domain
        for entry in method[1:]:
            if isinstance(entry, dict):
                if "Domain" in entry and domain == "Automatic":
                    method_domain = entry["Domain"]
                continue
            raise ValueError(f"unsupported method option payload: {entry!r}")
        return method_name, method_domain
    return method, domain


def TeukolskyPointParticleMode(
    s: int,
    ell: int,
    m: int,
    *args,
    domain="Automatic",
    source_type="Automatic",
    accelerator="cpu",
    device_id=0,
):
    if len(args) == 1:
        orbit = args[0]
        return solve_point_particle_mode(
            s=s, ell=ell, m=m, n=0, k=0, orbit=orbit,
            domain=domain, source_type=source_type, accelerator=accelerator, device_id=device_id,
        )

    if len(args) == 2:
        index, orbit = args
        if orbit.kind == "spherical":
            return solve_point_particle_mode(
                s=s, ell=ell, m=m, n=0, k=index, orbit=orbit,
                domain=domain, source_type=source_type, accelerator=accelerator, device_id=device_id,
            )
        if orbit.kind == "eccentric-equatorial":
            return solve_point_particle_mode(
                s=s, ell=ell, m=m, n=index, k=0, orbit=orbit,
                domain=domain, source_type=source_type, accelerator=accelerator, device_id=device_id,
            )
        raise ValueError("four-argument shorthand is only defined for spherical or eccentric-equatorial orbits")

    if len(args) == 3:
        n, k, orbit = args
        return solve_point_particle_mode(
            s=s, ell=ell, m=m, n=n, k=k, orbit=orbit,
            domain=domain, source_type=source_type, accelerator=accelerator, device_id=device_id,
        )

    raise TypeError("TeukolskyPointParticleMode expects (s,l,m,orbit), (s,l,m,index,orbit), or (s,l,m,n,k,orbit)")


def TeukolskyRadial(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    *,
    eigenvalue: complex | None = None,
    method: str = "NumericalIntegration",
    boundary_conditions: tuple[str, ...] = ("In", "Up"),
    domain="Automatic",
    renormalized_angular_momentum_value: complex | None = None,
):
    method_name, resolved_domain = _parse_radial_method(method, domain)
    return solve_radial(
        s=s,
        ell=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=eigenvalue,
        method=method_name,
        boundary_conditions=boundary_conditions,
        renormalized_angular_momentum_value=renormalized_angular_momentum_value,
        domain=None if resolved_domain == "Automatic" else resolved_domain,
    )


def TeukolskyRadialFunction(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    boundary_condition: str,
    *,
    eigenvalue: complex | None = None,
    method: str = "NumericalIntegration",
    domain="Automatic",
    renormalized_angular_momentum_value: complex | None = None,
):
    radial = TeukolskyRadial(
        s=s,
        ell=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=eigenvalue,
        method=method,
        domain=domain,
        boundary_conditions=(boundary_condition,),
        renormalized_angular_momentum_value=renormalized_angular_momentum_value,
    )
    if boundary_condition not in radial:
        raise ValueError(f"unsupported boundary condition: {boundary_condition!r}")
    return radial[boundary_condition]
