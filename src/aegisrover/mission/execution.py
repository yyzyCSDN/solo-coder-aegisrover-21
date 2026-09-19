"""Waypoint execution with geofence supervision.

Three failure modes are covered here. A concave geofence must be evaluated against
its real boundary — a sampling-only check between two points happily walks through
the notch of an L-shaped exclusion zone. A mission must not be reported as
completed while waypoints are still outstanding, even if the operator presses
"complete". And a crashed runner must not force the whole route to start over:
progress used to live only in memory, so a fault halfway through the route threw
away every waypoint already covered. :meth:`MissionExecution.checkpoint` captures
the runner state behind a digest and :meth:`MissionExecution.restore` brings it
back — but only against the exact plan that was checkpointed, so a mission whose
route was edited in the meantime is restarted explicitly instead of silently
flying the old line. Waypoints already consumed stay consumed, so resuming never
double-counts the ground already covered.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aegisrover.storage.repository import Repository, canonical_json

__all__ = ('Geofence', 'WaypointRunner', 'MissionExecution', 'ExecutionError',
           'ExecutionCheckpointStore', 'CHECKPOINT_NAMESPACE')

Point = tuple[float, float]

#: Repository namespace where durable execution checkpoints are stored.
CHECKPOINT_NAMESPACE = 'execution_checkpoints'

_CHECKPOINT_FIELDS = ('mission_id', 'state', 'index', 'skipped', 'events',
                      'abort_reason', 'waypoints', 'tolerance', 'plan_digest',
                      'sequence', 'digest')

_STATES = ('idle', 'running', 'paused', 'completed', 'aborted')


class ExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Geofence:
    """Closed polygon with a membership test that handles concave shapes."""

    polygon: tuple[Point, ...]
    margin: float = 0.0

    def __post_init__(self):
        if len(self.polygon) < 3:
            raise ExecutionError('a geofence needs at least three points')
        if self.margin < 0:
            raise ExecutionError('margin must not be negative')

    def contains(self, point: Point) -> bool:
        x, y = point
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < xi:
                    inside = not inside
        if inside:
            return True
        if self.margin > 0:
            return self._distance_to_boundary(point) <= self.margin
        return False

    def segment_inside(self, a: Point, b: Point) -> bool:
        """True when the whole segment stays inside the fence.

        Endpoints inside is not enough: a segment can leave through the notch of a
        concave polygon and come back. Every edge crossing is therefore checked
        against the fence boundary.
        """
        if not (self.contains(a) and self.contains(b)):
            return False
        edges = list(zip(self.polygon, self.polygon[1:] + (self.polygon[0],)))
        for p, q in edges:
            if _segments_cross(a, b, p, q):
                return False
        midpoint = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        return self.contains(midpoint)

    def violations(self, path: Sequence[Point]) -> list[tuple[int, Point]]:
        out: list[tuple[int, Point]] = []
        for index, (a, b) in enumerate(zip(path, path[1:])):
            if not self.segment_inside(a, b):
                out.append((index, b))
        return out

    def _distance_to_boundary(self, point: Point) -> float:
        best = math.inf
        n = len(self.polygon)
        for i in range(n):
            a = self.polygon[i]
            b = self.polygon[(i + 1) % n]
            best = min(best, _point_segment_distance(point, a, b))
        return best


@dataclass
class WaypointRunner:
    waypoints: Sequence[Point]
    tolerance: float = 0.5
    index: int = 0
    skipped: list[Point] = field(default_factory=list)

    def __post_init__(self):
        self.waypoints = tuple((float(x), float(y)) for x, y in self.waypoints)
        if self.tolerance <= 0:
            raise ExecutionError('tolerance must be positive')

    @property
    def current(self) -> Point | None:
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def advance(self, position: Point) -> int:
        while self.index < len(self.waypoints):
            target = self.waypoints[self.index]
            if math.dist(position, target) <= self.tolerance:
                self.index += 1
                continue
            for later in range(self.index + 1, len(self.waypoints)):
                if math.dist(position, self.waypoints[later]) <= self.tolerance:
                    self.skipped.extend(self.waypoints[self.index:later])
                    self.index = later + 1
                    break
            else:
                break
            continue
        return self.index

    def remaining_distance(self, position: Point) -> float:
        if self.done:
            return 0.0
        total = math.dist(position, self.waypoints[self.index])
        for a, b in zip(self.waypoints[self.index:], self.waypoints[self.index + 1:]):
            total += math.dist(a, b)
        return total

    def progress(self) -> float:
        if not self.waypoints:
            return 1.0
        return self.index / len(self.waypoints)


@dataclass
class MissionExecution:
    mission_id: str
    runner: WaypointRunner
    fence: Geofence | None = None
    state: str = 'idle'
    events: list[dict] = field(default_factory=list)
    abort_reason: str | None = None

    def start(self, position: Point) -> str:
        if self.state != 'idle':
            raise ExecutionError(f'cannot start from {self.state}')
        if not self.runner.waypoints:
            raise ExecutionError('mission has no waypoints')
        if self.fence is not None and not self.fence.contains(position):
            raise ExecutionError('start position is outside the geofence')
        if self.fence is not None:
            path = (position, *self.runner.waypoints)
            if self.fence.violations(path):
                raise ExecutionError('planned route leaves the geofence')
        self._log('start', {'position': position})
        self.state = 'running'
        return self.state

    def tick(self, position: Point) -> str:
        if self.state != 'running':
            return self.state
        if self.fence is not None and not self.fence.contains(position):
            self.abort('left the geofence')
            return self.state
        previous = self.runner.index
        self.runner.advance(position)
        if self.runner.index != previous:
            self._log('waypoint', {'index': self.runner.index, 'skipped': len(self.runner.skipped)})
        if self.runner.done:
            self.state = 'completed'
            self._log('complete', {'skipped': len(self.runner.skipped)})
        return self.state

    def pause(self) -> str:
        if self.state != 'running':
            raise ExecutionError(f'cannot pause from {self.state}')
        self.state = 'paused'
        self._log('pause', {})
        return self.state

    def resume(self) -> str:
        if self.state != 'paused':
            raise ExecutionError(f'cannot resume from {self.state}')
        self.state = 'running'
        self._log('resume', {})
        return self.state

    def complete(self) -> str:
        if self.state in ('completed', 'aborted'):
            return self.state
        if not self.runner.done:
            raise ExecutionError('waypoints still outstanding')
        self.state = 'completed'
        self._log('complete', {})
        return self.state

    def abort(self, reason: str) -> str:
        if self.state in ('completed', 'aborted'):
            return self.state
        self.state = 'aborted'
        self.abort_reason = reason
        self._log('abort', {'reason': reason})
        return self.state

    def summary(self) -> dict:
        return {'mission_id': self.mission_id, 'state': self.state,
                'progress': round(self.runner.progress(), 6),
                'skipped': len(self.runner.skipped), 'events': len(self.events),
                'abort_reason': self.abort_reason}

    # -- checkpoint / restore ----------------------------------------------------
    def checkpoint(self) -> dict:
        """Digest-protected snapshot of everything needed to resume this run.

        The snapshot pins the full waypoint plan and its digest so a restore can
        prove the route has not changed underneath it. ``sequence`` is the event
        count — it never decreases within a run — which lets a store reject a
        stale checkpoint instead of rolling progress backwards.
        """
        body = {
            'mission_id': self.mission_id,
            'state': self.state,
            'index': self.runner.index,
            'skipped': [list(p) for p in self.runner.skipped],
            'events': [dict(e) for e in self.events],
            'abort_reason': self.abort_reason,
            'waypoints': [list(w) for w in self.runner.waypoints],
            'tolerance': self.runner.tolerance,
            'sequence': len(self.events),
        }
        body['plan_digest'] = _plan_digest(self.runner.waypoints)
        return {**body, 'digest': _checkpoint_digest(body)}

    @staticmethod
    def verify_checkpoint(checkpoint: dict) -> bool:
        """True when the checkpoint carries every field and an intact digest."""
        if not all(field in checkpoint for field in _CHECKPOINT_FIELDS):
            return False
        body = {k: v for k, v in checkpoint.items() if k != 'digest'}
        return _checkpoint_digest(body) == checkpoint['digest']

    @classmethod
    def restore(cls, checkpoint: dict, *, waypoints: Sequence[Point] | None = None,
                fence: 'Geofence | None' = None) -> 'MissionExecution':
        """Rebuild an execution from :meth:`checkpoint`.

        ``waypoints`` is the plan the mission holds *now*; when given, it must be
        identical to the checkpointed plan. A mismatch means the route was edited
        after the crash and the caller must restart explicitly — resuming onto a
        different route would leave the flown trajectory disagreeing with the
        plan. A runner checkpointed while ``running`` comes back ``paused`` so
        resuming is an explicit decision.
        """
        if not isinstance(checkpoint, dict) or not cls.verify_checkpoint(checkpoint):
            raise ExecutionError('checkpoint is missing fields or fails its digest')
        if checkpoint['state'] not in _STATES:
            raise ExecutionError(f"checkpoint has unknown state {checkpoint['state']!r}")
        plan = tuple((float(x), float(y)) for x, y in checkpoint['waypoints'])
        if _plan_digest(plan) != checkpoint['plan_digest']:
            raise ExecutionError('checkpoint plan fails its digest')
        if waypoints is not None and _plan_digest(waypoints) != checkpoint['plan_digest']:
            raise ExecutionError('mission plan changed since the checkpoint; restart required')
        index = int(checkpoint['index'])
        if not 0 <= index <= len(plan):
            raise ExecutionError('checkpoint index is outside the plan')
        runner = WaypointRunner(
            plan,
            tolerance=float(checkpoint['tolerance']),
            index=index,
            skipped=[(float(x), float(y)) for x, y in checkpoint['skipped']],
        )
        state = 'paused' if checkpoint['state'] == 'running' else checkpoint['state']
        return cls(checkpoint['mission_id'], runner, fence, state=state,
                   events=[dict(e) for e in checkpoint['events']],
                   abort_reason=checkpoint['abort_reason'])

    def _log(self, kind: str, payload: dict) -> None:
        self.events.append({'event': kind, **payload})


class ExecutionCheckpointStore:
    """Durable per-mission checkpoints on top of the versioned repository.

    Saving is monotonic in the checkpoint sequence: an older snapshot (taken
    before the latest save, e.g. flushed late by a dying process) is rejected
    rather than allowed to roll the mission backwards.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def save(self, execution: MissionExecution) -> dict:
        checkpoint = execution.checkpoint()
        existing = self._repo.maybe_get(CHECKPOINT_NAMESPACE, execution.mission_id)
        if existing is not None and existing.payload['sequence'] > checkpoint['sequence']:
            raise ExecutionError('refusing to overwrite a newer checkpoint')
        self._repo.put(CHECKPOINT_NAMESPACE, execution.mission_id, checkpoint)
        return checkpoint

    def latest(self, mission_id: str) -> dict | None:
        record = self._repo.maybe_get(CHECKPOINT_NAMESPACE, mission_id)
        return None if record is None else dict(record.payload)

    def restore(self, mission_id: str, *, waypoints: Sequence[Point] | None = None,
                fence: 'Geofence | None' = None) -> MissionExecution | None:
        """Resume ``mission_id`` from its latest checkpoint, or None if there is none."""
        checkpoint = self.latest(mission_id)
        if checkpoint is None:
            return None
        return MissionExecution.restore(checkpoint, waypoints=waypoints, fence=fence)

    def discard(self, mission_id: str) -> None:
        """Drop the checkpoint once the mission is finished with it."""
        if self._repo.maybe_get(CHECKPOINT_NAMESPACE, mission_id) is not None:
            self._repo.delete(CHECKPOINT_NAMESPACE, mission_id)


def _plan_digest(waypoints: Sequence[Point]) -> str:
    body = canonical_json([[float(x), float(y)] for x, y in waypoints])
    return hashlib.sha256(body.encode()).hexdigest()


def _checkpoint_digest(body: dict) -> str:
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def _point_segment_distance(point: Point, a: Point, b: Point) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = point[0] - a[0], point[1] - a[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    return math.hypot(wx - t * vx, wy - t * vy)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if (o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d)):
        return True
    if (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)
