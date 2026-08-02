"""Coordinate transforms for the fixed Beihu build-bound world.

Three external conventions meet at the navigation boundary:

- GPS (WGS84 lat/lon degrees) from SCADA pose and ``Get_Route`` waypoints;
- Unity world X/Z (left-handed, Y-up) used by the extracted water mesh,
  fixed buoys and route.txt positions;
- the internal ``map`` frame: right-handed ENU metres with yaw measured
  counter-clockwise from +x (East), used by the planner and dynamics.

This module freezes the conversions without guessing:

- GPS <-> ENU uses a local equirectangular projection around the first
  calibration anchor from the build-bound ``GpsSettings``;
- Unity X/Z <-> ENU historically used candidate transforms fitted from the
  same two anchors (``fit_unity_to_enu``/``fit_affine_unity_to_enu``).
  Live evidence (2026-07-30) shows the build's GPS converter does NOT follow
  those anchors: the route GPS column and the ship GPS are instead an exact
  affine image of the Unity coordinates.  The supervised entry therefore
  fits ``AffineTransform2D`` from the route waypoints at startup
  (``fit_route_converter``) and gates on residual exactness plus an
  independent ship-in-water check; the anchor-fitted candidates remain for
  offline/analysis use.

Heading conversions follow the evidence that SCADA ``pose.yaw`` is a compass
heading (North = 0, clockwise positive, degrees) while the planner uses a
right-handed math yaw (East = 0, counter-clockwise positive, radians).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import Sequence

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_ROUTE_TOLERANCE_M = 0.5


def _wrap_pi(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class GpsProjector:
    """Local tangent-plane projection around a fixed GPS origin."""

    origin_lat_deg: float
    origin_lon_deg: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.origin_lat_deg <= 90.0 and -180.0 <= self.origin_lon_deg <= 180.0):
            raise ValueError("gps origin is out of range")

    def gps_to_enu(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        lat = radians(lat_deg)
        lon = radians(lon_deg)
        lat0 = radians(self.origin_lat_deg)
        lon0 = radians(self.origin_lon_deg)
        x = (lon - lon0) * cos(lat0) * EARTH_RADIUS_M
        y = (lat - lat0) * EARTH_RADIUS_M
        return x, y

    def enu_to_gps(self, x_m: float, y_m: float) -> tuple[float, float]:
        lat0 = radians(self.origin_lat_deg)
        lon0 = radians(self.origin_lon_deg)
        lat = lat0 + y_m / EARTH_RADIUS_M
        lon = lon0 + x_m / (EARTH_RADIUS_M * cos(lat0))
        return degrees(lat), degrees(lon)


@dataclass(frozen=True)
class SimilarityTransform2D:
    """Unity X/Z -> ENU similarity transform (rotation, uniform scale, translation).

    ``reflected=True`` first flips the Unity z axis before rotating, which is
    the candidate chirality when a left-handed-to-right-handed reflection is
    required.  Chirality is never assumed: it must be selected by independent
    waypoint residuals at runtime.
    """

    scale: float
    rotation_rad: float
    tx: float
    ty: float
    reflected: bool
    version: str = "unity-enu-similarity-v1"

    def unity_to_enu(self, ux: float, uz: float) -> tuple[float, float]:
        z = -uz if self.reflected else uz
        c = cos(self.rotation_rad)
        s = sin(self.rotation_rad)
        x = self.scale * (c * ux - s * z) + self.tx
        y = self.scale * (s * ux + c * z) + self.ty
        return x, y

    def enu_to_unity(self, x: float, y: float) -> tuple[float, float]:
        c = cos(self.rotation_rad)
        s = sin(self.rotation_rad)
        dx = (x - self.tx) / self.scale
        dy = (y - self.ty) / self.scale
        ux = c * dx + s * dy
        z = -s * dx + c * dy
        return ux, (-z if self.reflected else z)


def fit_unity_to_enu(
    anchors: dict,
    projector: GpsProjector,
    *,
    reflected: bool,
) -> SimilarityTransform2D:
    """Fit the Unity->ENU similarity transform from the two GpsSettings anchors.

    The two anchors determine a similarity transform exactly, so this fit has
    no residual by construction; independent verification is mandatory and
    provided by :func:`verify_route_points`.
    """

    required = (
        "latitude1",
        "longitude1",
        "unity_x1",
        "unity_z1",
        "latitude2",
        "longitude2",
        "unity_x2",
        "unity_z2",
    )
    if any(key not in anchors for key in required):
        raise ValueError("gps anchors are incomplete")
    e1 = projector.gps_to_enu(float(anchors["latitude1"]), float(anchors["longitude1"]))
    e2 = projector.gps_to_enu(float(anchors["latitude2"]), float(anchors["longitude2"]))
    u1 = (float(anchors["unity_x1"]), float(anchors["unity_z1"]))
    u2 = (float(anchors["unity_x2"]), float(anchors["unity_z2"]))

    def _flip(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0], -point[1]) if reflected else point

    f1 = _flip(u1)
    f2 = _flip(u2)
    du = (f2[0] - f1[0], f2[1] - f1[1])
    de = (e2[0] - e1[0], e2[1] - e1[1])
    len_u = hypot(*du)
    len_e = hypot(*de)
    if len_u < 1e-9 or len_e < 1e-9:
        raise ValueError("anchor pairs are degenerate")
    scale = len_e / len_u
    rotation = atan2(de[1], de[0]) - atan2(du[1], du[0])
    c = cos(rotation)
    s = sin(rotation)
    tx = e1[0] - scale * (c * f1[0] - s * f1[1])
    ty = e1[1] - scale * (s * f1[0] + c * f1[1])
    return SimilarityTransform2D(scale=scale, rotation_rad=rotation, tx=tx, ty=ty, reflected=reflected)


@dataclass(frozen=True)
class AffineAxisAlignedTransform:
    """Unity X/Z -> ENU axis-aligned affine transform (no rotation, per-axis scale).

    This mirrors the most likely ``GPSConverter2`` implementation
    (``ux = a*lon + b``, ``uz = c*lat + d`` fitted from two anchors).  The
    non-uniform scale absorbs the anchors' measurement noise, so it must still
    pass independent waypoint residual verification before use.
    """

    scale_x: float
    scale_z: float
    tx: float
    ty: float
    projector: GpsProjector
    version: str = "unity-enu-axis-affine-v1"

    def unity_to_enu(self, ux: float, uz: float) -> tuple[float, float]:
        return self.scale_x * ux + self.tx, self.scale_z * uz + self.ty

    def enu_to_unity(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.tx) / self.scale_x, (y - self.ty) / self.scale_z


def fit_affine_unity_to_enu(anchors: dict, projector: GpsProjector) -> AffineAxisAlignedTransform:
    """Fit the axis-aligned Unity<->GPS model and express it in ENU metres."""

    required = (
        "latitude1",
        "longitude1",
        "unity_x1",
        "unity_z1",
        "latitude2",
        "longitude2",
        "unity_x2",
        "unity_z2",
    )
    if any(key not in anchors for key in required):
        raise ValueError("gps anchors are incomplete")
    lat1, lon1 = float(anchors["latitude1"]), float(anchors["longitude1"])
    lat2, lon2 = float(anchors["latitude2"]), float(anchors["longitude2"])
    ux1, uz1 = float(anchors["unity_x1"]), float(anchors["unity_z1"])
    ux2, uz2 = float(anchors["unity_x2"]), float(anchors["unity_z2"])
    if abs(lon2 - lon1) < 1e-12 or abs(lat2 - lat1) < 1e-12:
        raise ValueError("anchor pairs are degenerate")
    a = (ux2 - ux1) / (lon2 - lon1)
    b = ux1 - a * lon1
    c = (uz2 - uz1) / (lat2 - lat1)
    d = uz1 - c * lat1
    e1 = projector.gps_to_enu(lat1, lon1)
    e2 = projector.gps_to_enu(lat2, lon2)
    scale_x = (e2[0] - e1[0]) / (ux2 - ux1)
    scale_z = (e2[1] - e1[1]) / (uz2 - uz1)
    return AffineAxisAlignedTransform(
        scale_x=scale_x,
        scale_z=scale_z,
        tx=e1[0] - scale_x * ux1,
        ty=e1[1] - scale_z * uz1,
        projector=projector,
    )


@dataclass(frozen=True)
class RouteTransformVerification:
    reflected: bool
    max_residual_m: float
    mean_residual_m: float
    residuals_m: tuple[float, ...]
    passed: bool
    tolerance_m: float


def verify_route_points(
    transform: SimilarityTransform2D | AffineAxisAlignedTransform,
    unity_points: Sequence[tuple[float, float]],
    gps_points: Sequence[tuple[float, float]],
    projector: GpsProjector,
    *,
    tolerance_m: float = DEFAULT_ROUTE_TOLERANCE_M,
) -> RouteTransformVerification:
    """Compare sidecar Unity waypoint positions with GPS waypoints from Get_Route.

    ``unity_points`` come from the build-bound sidecar, ``gps_points`` from the
    live route service.  They must describe the same waypoints in the same
    order; the caller is responsible for route identity/version checks first.
    """

    if len(unity_points) != len(gps_points) or not unity_points:
        raise ValueError("unity and gps waypoint lists must align and be non-empty")
    residuals = []
    for (ux, uz), (lat, lon) in zip(unity_points, gps_points):
        expected = transform.unity_to_enu(ux, uz)
        measured = projector.gps_to_enu(lat, lon)
        residuals.append(hypot(expected[0] - measured[0], expected[1] - measured[1]))
    max_residual = max(residuals)
    mean_residual = sum(residuals) / len(residuals)
    return RouteTransformVerification(
        reflected=getattr(transform, "reflected", False),
        max_residual_m=max_residual,
        mean_residual_m=mean_residual,
        residuals_m=tuple(residuals),
        passed=max_residual <= tolerance_m,
        tolerance_m=tolerance_m,
    )


@dataclass(frozen=True)
class AffineTransform2D:
    """General 2-D affine Unity X/Z -> ENU: x = a*ux + b*uz + tx, y = c*ux + d*uz + ty.

    Live evidence (2026-07-30, docs/evidence/P0-18): the GPS converter in the
    running build does not follow the ``GpsSettings`` anchors — both the route
    GPS column and the ship's own GPS are an exact affine image of the Unity
    coordinates, so the converter is fitted from the route waypoints at
    startup instead of being assumed from setting.txt.
    """

    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float
    version: str = "unity-enu-route-fitted-affine-v1"

    def unity_to_enu(self, ux: float, uz: float) -> tuple[float, float]:
        return self.a * ux + self.b * uz + self.tx, self.c * ux + self.d * uz + self.ty

    def enu_to_unity(self, x: float, y: float) -> tuple[float, float]:
        det = self.a * self.d - self.b * self.c
        if abs(det) < 1e-12:
            raise ValueError("affine transform is singular")
        ix, iy = x - self.tx, y - self.ty
        return (self.d * ix - self.b * iy) / det, (-self.c * ix + self.a * iy) / det

    @property
    def reflected(self) -> bool:
        return (self.a * self.d - self.b * self.c) < 0.0

    def coefficients(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.tx, self.ty)

    def singular_values(self) -> tuple[float, float]:
        """Largest/smallest singular value of the linear part (scale band)."""

        m11 = self.a * self.a + self.c * self.c
        m12 = self.a * self.b + self.c * self.d
        m22 = self.b * self.b + self.d * self.d
        trace = m11 + m22
        det = m11 * m22 - m12 * m12
        disc = max(0.0, trace * trace / 4.0 - det) ** 0.5
        return (max(0.0, trace / 2.0 + disc)) ** 0.5, (max(0.0, trace / 2.0 - disc)) ** 0.5

    def max_scale(self) -> float:
        return self.singular_values()[0]


def fit_route_converter(
    unity_points: Sequence[tuple[float, float]],
    enu_points: Sequence[tuple[float, float]],
) -> tuple[AffineTransform2D, tuple[float, ...]]:
    """Least-squares general affine Unity->ENU from corresponding waypoints.

    Returns the transform and per-point residuals.  A two-anchor GPS
    converter is exactly affine, so a consistent route yields sub-metre
    residuals; anything larger means the live route is not the build-bound
    one (wrong scene, reordered, corrupted) and the caller must fail closed.
    """

    if len(unity_points) != len(enu_points) or len(unity_points) < 3:
        raise ValueError("route converter fit needs >= 3 aligned waypoint pairs")
    import numpy as np

    rows = []
    values = []
    for (ux, uz), (ex, ey) in zip(unity_points, enu_points):
        rows.append([ux, uz, 1.0, 0.0, 0.0, 0.0])
        values.append(ex)
        rows.append([0.0, 0.0, 0.0, ux, uz, 1.0])
        values.append(ey)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)
    sol_a, sol_b, sol_tx, sol_c, sol_d, sol_ty = (float(v) for v in solution)
    transform = AffineTransform2D(a=sol_a, b=sol_b, c=sol_c, d=sol_d, tx=sol_tx, ty=sol_ty)
    residuals = tuple(
        hypot(transform.unity_to_enu(ux, uz)[0] - ex, transform.unity_to_enu(ux, uz)[1] - ey)
        for (ux, uz), (ex, ey) in zip(unity_points, enu_points)
    )
    return transform, residuals


def compass_yaw_deg_to_math_yaw_rad(compass_deg: float) -> float:
    """SCADA compass heading (North=0, clockwise+) -> math yaw (East=0, CCW+)."""

    return _wrap_pi(pi / 2.0 - radians(compass_deg))

def math_yaw_rad_to_compass_deg(yaw_rad: float) -> float:
    """Math yaw -> SCADA convention compass heading in degrees [-180, 180)."""

    compass = degrees(pi / 2.0 - yaw_rad)
    return ((compass + 180.0) % 360.0) - 180.0


def compass_yaw_rate_degs_to_math_rad_s(rotate_speed_degs: float) -> float:
    """SCADA rotate_speed (deg/s, compass clockwise+) -> math yaw rate (rad/s)."""

    return -radians(rotate_speed_degs)
