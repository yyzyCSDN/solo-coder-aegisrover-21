"""Crash-safe mission recovery: resume a mission from its interruption point.

A fault halfway through a route must not force the mission to restart. Three
guarantees make a resumed run indistinguishable, for counting purposes, from a
run that never stopped:

* **Same plan.** Every checkpoint carries a digest of the planned route, so a
  resume against a different (or tampered) plan is refused instead of silently
  following the wrong trajectory.
* **No double counting of distance.** Covered distance is measured along the
  plan by the furthest arc length reached. The robot is usually towed or drives
  back to the interruption point after a fault; every metre of that repeat
  traversal lands on arc length that was already covered and adds nothing.
* **No double counting of samples.** Collected data is recorded under a stable
  idempotency key (sample id, else a spatial key along the plan). Re-recording
  the same sample after the restart is a no-op.

The checkpoint is plain data with a checksum, so it can be written to the
repository or a file by the caller; this module never assumes it survives in
memory.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from aegisrover.storage.repository import canonical_json

__all__ = (
    'RecoveryError', 'RouteMismatch', 'PlanDrift', 'PlannedRoute',
    'RouteProjection', 'ProgressLedger', 'MissionCheckpoint', 'ResumableMission',
)

Point = tuple[float, float]

_EPS = 1e-9


class RecoveryError(RuntimeError):
    pass


class RouteMismatch(RecoveryError):
    """The checkpoint belongs to a different plan than the one being resumed."""


class PlanDrift(RecoveryError):
    """Resume position is too far from the planned trajectory to project."""


@dataclass(frozen=True)
class RouteProjection:
    """A position expressed as an arc-length coordinate on the plan."""

    arc: float
    point: Point
    cross_track: float
    segment: int


class PlannedRoute:
    """Polyline plan parameterised by accumulated arc length.

    Arc length is the coordinate every other guarantee hangs off: it orders the
    route, locates the interruption point, and measures covered distance in a
    way that ignores backtracking along an already covered stretch.
    """

    def __init__(self, points: Sequence[Point], *, cross_track_tolerance: float = 1.0,
                 arc_tolerance: float = 0.5):
        points = tuple((float(x), float(y)) for x, y in points)
        if len(points) < 2:
            raise RecoveryError('a route needs at least two points')
        if cross_track_tolerance <= 0:
            raise RecoveryError('cross_track_tolerance must be positive')
        if arc_tolerance < 0:
            raise RecoveryError('arc_tolerance must not be negative')
        self.points = points
        self.cross_track_tolerance = float(cross_track_tolerance)
        self.arc_tolerance = float(arc_tolerance)
        edges = [0.0]
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            if length <= _EPS:
                raise RecoveryError('duplicate consecutive route points')
            edges.append(edges[-1] + length)
        self.edges = tuple(edges)
        self.total_length = self.edges[-1]
        self.digest = self._digest(points, self.cross_track_tolerance)

    @staticmethod
    def _digest(points: Sequence[Point], tolerance: float) -> str:
        material = canonical_json({'points': points, 'cross_track_tolerance': tolerance})
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def point_at(self, arc: float) -> Point:
        arc = max(0.0, min(self.total_length, arc))
        for i in range(len(self.points) - 1):
            start, end = self.edges[i], self.edges[i + 1]
            if arc <= end + _EPS:
                t = 0.0 if end <= start else (arc - start) / (end - start)
                a, b = self.points[i], self.points[i + 1]
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return self.points[-1]

    def project(self, position: Point, *, enforce_tolerance: bool = True) -> RouteProjection:
        """Nearest point on the plan for ``position``.

        A robot resuming from a repair bay sits well off the route; the nearest
        point on the polyline is still its interruption coordinate, but its
        cross-track distance is large. Such a jump is legitimate *before* the
        robot has rejoined the plan; once it reports a position while running it
        must stay within tolerance, hence ``enforce_tolerance``.
        """
        best: RouteProjection | None = None
        px, py = float(position[0]), float(position[1])
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            vx, vy = b[0] - a[0], b[1] - a[1]
            length_sq = vx * vx + vy * vy
            t = max(0.0, min(1.0, ((px - a[0]) * vx + (py - a[1]) * vy) / length_sq))
            qx, qy = a[0] + t * vx, a[1] + t * vy
            cross = math.hypot(px - qx, py - qy)
            arc = self.edges[i] + t * math.sqrt(length_sq)
            if best is None or cross < best.cross_track:
                best = RouteProjection(arc, (qx, qy), cross, i)
        assert best is not None
        if enforce_tolerance and best.cross_track > self.cross_track_tolerance + _EPS:
            raise PlanDrift(
                f'position {position!r} is {best.cross_track:.3f} m off the plan '
                f'(tolerance {self.cross_track_tolerance:.3f})')
        return best

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class ProgressLedger:
    """Idempotent accounting of covered distance and collected samples.

    ``furthest_arc`` only moves forward, so driving back over ground that was
    already covered (the tow to the depot, the drive back to the breakpoint)
    cannot add distance. ``samples`` is a dict keyed by a stable id, so the same
    sample re-delivered after a restart is counted once.
    """

    furthest_arc: float = 0.0
    samples: dict[str, dict] = field(default_factory=dict)

    def record_position(self, arc: float) -> bool:
        """Advance the high-water mark. Returns True when new ground was covered."""
        if arc > self.furthest_arc + _EPS:
            self.furthest_arc = arc
            return True
        return False

    def add_sample(self, sample: dict, *, route: PlannedRoute | None = None) -> bool:
        """Add a sample under its idempotency key. Returns True if it was new."""
        key = sample_key(sample, route=route)
        if key in self.samples:
            return False
        self.samples[key] = dict(sample)
        return True

    def has_sample(self, key: str) -> bool:
        return key in self.samples

    def to_dict(self) -> dict:
        return {'furthest_arc': self.furthest_arc,
                'samples': [dict(sample, _key=key) for key, sample in sorted(self.samples.items())]}

    @staticmethod
    def from_dict(payload: dict) -> 'ProgressLedger':
        ledger = ProgressLedger(furthest_arc=float(payload.get('furthest_arc', 0.0)))
        for item in payload.get('samples', ()):
            sample = dict(item)
            key = sample.pop('_key')
            ledger.samples[key] = sample
        return ledger


def sample_key(sample: dict, *, route: PlannedRoute | None = None) -> str:
    """Stable identity for a sample.

    An explicit ``sample_id`` always wins; otherwise the sample's position is
    snapped to a 1 cm grid along the plan so two passes over the same physical
    observation point produce the same key.
    """
    sample_id = sample.get('sample_id') or sample.get('id')
    if sample_id is not None:
        return f'id:{sample_id}'
    x, y = sample.get('x'), sample.get('y')
    if x is None or y is None:
        raise RecoveryError('a sample needs a sample_id or an x/y position')
    if route is not None:
        projection = route.project((float(x), float(y)), enforce_tolerance=False)
        return f'arc:{projection.arc:.2f}'
    return f'pos:{float(x):.2f},{float(y):.2f}'


@dataclass(frozen=True)
class MissionCheckpoint:
    """Serializable interruption state, protected by a checksum.

    The digest binds the mission id, the plan digest and all progress, so
    corruption or hand-editing is detected on resume rather than applied.
    """

    mission_id: str
    plan_digest: str
    state: str
    resume_arc: float
    ledger: dict
    events: tuple[dict, ...] = ()
    revision: int = 1
    digest: str = ''

    def to_dict(self) -> dict:
        return {'mission_id': self.mission_id, 'plan_digest': self.plan_digest,
                'state': self.state, 'resume_arc': self.resume_arc,
                'ledger': self.ledger, 'events': [dict(e) for e in self.events],
                'revision': self.revision, 'digest': self.digest}

    @staticmethod
    def from_dict(payload: dict) -> 'MissionCheckpoint':
        cp = MissionCheckpoint(
            mission_id=payload['mission_id'],
            plan_digest=payload['plan_digest'],
            state=payload['state'],
            resume_arc=float(payload['resume_arc']),
            ledger=dict(payload['ledger']),
            events=tuple(payload.get('events', ())),
            revision=int(payload.get('revision', 1)),
            digest=payload['digest'],
        )
        if not cp.verify():
            raise RecoveryError('checkpoint checksum mismatch')
        return cp

    def verify(self) -> bool:
        return self.digest == self.compute_digest()

    def compute_digest(self) -> str:
        material = canonical_json({
            'mission_id': self.mission_id, 'plan_digest': self.plan_digest,
            'state': self.state, 'resume_arc': self.resume_arc,
            'ledger': self.ledger, 'events': list(self.events), 'revision': self.revision,
        })
        return hashlib.sha256(material.encode()).hexdigest()


#: state from which samples may be recorded
_RUNNING = 'running'


class ResumableMission:
    """Mission runner that survives a mid-run fault and resumes exactly once.

    Typical lifecycle::

        mission = ResumableMission.start('m1', route, start_position)
        mission.update(position)            # tick while running
        mission.collect({'sample_id': 's1', 'x': .., 'y': ..})
        checkpoint = mission.fault('power brownout').checkpoint()
        ... persist checkpoint, process restarts ...
        resumed = ResumableMission.restore(route, checkpoint)
        resumed.rejoin(position_at_depot)    # tow back, not counted
        resumed.resume(position_on_plan)
        resumed.update(position)             # continues the same accounting
    """

    def __init__(self, mission_id: str, route: PlannedRoute, *,
                 ledger: ProgressLedger | None = None, state: str = 'ready',
                 events: Iterable[dict] | None = None, revision: int = 1):
        self.mission_id = mission_id
        self.route = route
        self.ledger = ledger or ProgressLedger()
        self.state = state
        self.events: list[dict] = list(events or [])
        self.revision = revision
        self.last_arc: float | None = None

    # -- construction / recovery ----------------------------------------------
    @classmethod
    def start(cls, mission_id: str, route: PlannedRoute, position: Point) -> 'ResumableMission':
        mission = cls(mission_id, route)
        projection = route.project(position)
        if projection.arc > _EPS:
            raise RecoveryError('a new mission must start at the first route point')
        mission.state = 'running'
        mission.last_arc = 0.0
        mission.ledger.record_position(0.0)
        mission._log('start', {'arc': 0.0})
        return mission

    def checkpoint(self) -> MissionCheckpoint:
        resume_arc = self.ledger.furthest_arc
        cp = MissionCheckpoint(
            mission_id=self.mission_id,
            plan_digest=self.route.digest,
            state=self.state,
            resume_arc=resume_arc,
            ledger=self.ledger.to_dict(),
            events=tuple(self.events),
            revision=self.revision,
        )
        return replace(cp, digest=cp.compute_digest())

    @classmethod
    def restore(cls, route: PlannedRoute, checkpoint: MissionCheckpoint | dict) -> 'ResumableMission':
        cp = (checkpoint if isinstance(checkpoint, MissionCheckpoint)
              else MissionCheckpoint.from_dict(checkpoint))
        if cp.plan_digest != route.digest:
            raise RouteMismatch(
                f'checkpoint plan {cp.plan_digest!r} does not match route {route.digest!r}')
        if cp.state != 'faulted':
            raise RecoveryError(f'can only restore a faulted mission, not {cp.state!r}')
        mission = cls(cp.mission_id, route, ledger=ProgressLedger.from_dict(cp.ledger),
                      state='faulted', events=cp.events, revision=cp.revision)
        mission.last_arc = cp.resume_arc
        return mission

    def resume(self, position: Point) -> str:
        """Rejoin the plan and continue.

        The interruption point is the high-water mark recorded before the fault.
        The robot may physically stand before that point (still driving back) or
        at it; in both cases nothing is added until it passes the mark. A
        position already beyond the mark is impossible without having run the
        intervening ground and is rejected as a gap.
        """
        if self.state != 'faulted':
            raise RecoveryError(f'cannot resume from {self.state!r}')
        projection = self.route.project(position)
        if projection.arc > self.ledger.furthest_arc + self.route.arc_tolerance + _EPS:
            raise PlanDrift(
                f'resume position at arc {projection.arc:.3f} is ahead of the '
                f'interruption point {self.ledger.furthest_arc:.3f}')
        self.last_arc = projection.arc
        self.state = 'running'
        self.revision += 1
        self._log('resume', {'arc': projection.arc,
                             'resume_arc': self.ledger.furthest_arc})
        return self.state

    # -- runtime ---------------------------------------------------------------
    def update(self, position: Point) -> float:
        """Report a position; returns newly covered distance since the fault-free start.

        Positions behind the high-water mark cover no new ground (that is the
        repeat-traversal rule); the return value reflects only this tick's
        advance of the mark.
        """
        if self.state != 'running':
            raise RecoveryError(f'cannot update while {self.state!r}')
        projection = self.route.project(position)
        previous = self.ledger.furthest_arc
        self.ledger.record_position(projection.arc)
        self.last_arc = projection.arc
        if self.ledger.furthest_arc >= self.route.total_length - _EPS:
            self.ledger.furthest_arc = self.route.total_length
            self.state = 'completed'
            self.revision += 1
            self._log('complete', {'arc': self.route.total_length})
        return max(0.0, self.ledger.furthest_arc - previous)

    def collect(self, sample: dict) -> bool:
        """Record collected data; a repeated sample (post-restart) is ignored."""
        if self.state != _RUNNING:
            raise RecoveryError(f'cannot collect samples while {self.state!r}')
        is_new = self.ledger.add_sample(sample, route=self.route)
        if is_new:
            self._log('sample', {'key': sample_key(sample, route=self.route)})
        return is_new

    def fault(self, reason: str) -> 'ResumableMission':
        if self.state not in ('running', 'faulted'):
            raise RecoveryError(f'cannot fault from {self.state!r}')
        if self.state == 'running':
            self.state = 'faulted'
            self.revision += 1
            self._log('fault', {'reason': reason, 'resume_arc': self.ledger.furthest_arc})
        return self

    def rejoin(self, position: Point) -> float:
        """Move while faulted (tow truck, manual drive) without counting distance."""
        if self.state != 'faulted':
            raise RecoveryError(f'cannot rejoin while {self.state!r}')
        projection = self.route.project(position, enforce_tolerance=False)
        self.last_arc = projection.arc
        return self.ledger.furthest_arc

    # -- introspection ---------------------------------------------------------
    @property
    def progress(self) -> float:
        return self.ledger.furthest_arc / self.route.total_length

    @property
    def covered_distance(self) -> float:
        return self.ledger.furthest_arc

    @property
    def sample_count(self) -> int:
        return len(self.ledger.samples)

    def summary(self) -> dict:
        return {'mission_id': self.mission_id, 'state': self.state,
                'plan_digest': self.route.digest, 'revision': self.revision,
                'covered_distance': round(self.ledger.furthest_arc, 6),
                'total_distance': round(self.route.total_length, 6),
                'progress': round(self.progress, 6),
                'samples': len(self.ledger.samples)}

    def _log(self, kind: str, payload: dict) -> None:
        self.events.append({'event': kind, **payload})
