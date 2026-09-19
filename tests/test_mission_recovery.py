"""Acceptance tests for crash-safe mission recovery.

The failure mode: a mission faults halfway, the whole run restarts, and every
metre and every sample already taken gets counted a second time. Recovery must
instead resume from the interruption point, never repeat a count, and only
resume against the original plan.
"""
import pytest

from aegisrover.mission.recovery import (
    MissionCheckpoint,
    PlannedRoute,
    PlanDrift,
    ProgressLedger,
    RecoveryError,
    ResumableMission,
    RouteMismatch,
)

ROUTE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]  # 20 m L-shape


def test_resume_after_fault_does_not_repeat_traversed_distance():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m1', route, (0.0, 0.0))

    first_leg = mission.update((4.0, 0.0))           # 4 m covered, then fault
    mission.collect({'sample_id': 's1', 'x': 2.0, 'y': 0.0})
    mission.fault('power brownout')
    payload = mission.checkpoint().to_dict()         # persisted across restart

    # New process: robot was towed 4 m back to the depot, off the plan.
    route2 = PlannedRoute(ROUTE)
    resumed = ResumableMission.restore(route2, MissionCheckpoint.from_dict(payload))
    resumed.rejoin((-5.0, 0.0))                        # tow, must not count
    resumed.resume((4.0, 0.0))                         # back at interruption point

    second_leg = 0.0
    second_leg += resumed.update((10.0, 0.0))
    second_leg += resumed.update((10.0, 10.0))

    assert resumed.state == 'completed'
    # first 4 + 6 + 10 == full 20 m; the 4 m tow and drive-back add nothing
    assert first_leg + second_leg == pytest.approx(20.0)
    assert resumed.covered_distance == pytest.approx(20.0)
    assert resumed.progress == pytest.approx(1.0)


def test_samples_collected_before_fault_are_not_double_counted():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m2', route, (0.0, 0.0))
    mission.update((5.0, 0.0))
    assert mission.collect({'sample_id': 's1', 'kind': 'reading', 'x': 3.0, 'y': 0.0})
    assert mission.collect({'sample_id': 's2', 'kind': 'reading', 'x': 5.0, 'y': 0.0})
    mission.fault('comms loss')
    resumed = ResumableMission.restore(PlannedRoute(ROUTE), mission.checkpoint())
    resumed.resume((5.0, 0.0))

    # Same sample redelivered by the sensor pipeline after restart: idempotent.
    assert resumed.collect({'sample_id': 's1', 'kind': 'reading', 'x': 3.0, 'y': 0.0}) is False
    assert resumed.collect({'sample_id': 's3', 'kind': 'reading', 'x': 8.0, 'y': 0.0}) is True
    assert resumed.sample_count == 3
    # id-based samples keep their explicit keys; nothing was re-counted.
    assert resumed.ledger.has_sample('id:s1') and resumed.ledger.has_sample('id:s2')


def test_resume_is_refused_against_a_different_plan():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m3', route, (0.0, 0.0))
    mission.update((6.0, 0.0))
    mission.fault('fault')
    checkpoint = mission.checkpoint()

    other_plan = PlannedRoute([(0.0, 0.0), (10.0, 0.0), (10.0, 11.0)])
    with pytest.raises(RouteMismatch):
        ResumableMission.restore(other_plan, checkpoint)


def test_tampered_checkpoint_is_detected():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m4', route, (0.0, 0.0))
    mission.update((6.0, 0.0))
    mission.fault('fault')
    payload = mission.checkpoint().to_dict()
    payload['resume_arc'] = 9.0                       # hand-edited progress
    with pytest.raises(RecoveryError):
        MissionCheckpoint.from_dict(payload)
    payload = mission.checkpoint().to_dict()
    payload['ledger']['furthest_arc'] = 9.0
    with pytest.raises(RecoveryError):
        MissionCheckpoint.from_dict(payload)


def test_cannot_resume_ahead_of_the_breakpoint():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m5', route, (0.0, 0.0))
    mission.update((4.0, 0.0))
    mission.fault('fault')
    resumed = ResumableMission.restore(PlannedRoute(ROUTE), mission.checkpoint())
    # Claiming to be at 8 m while the interruption point is at 4 m skips ground.
    with pytest.raises(PlanDrift):
        resumed.resume((8.0, 0.0))
    # And a position nowhere near the plan cannot be a resume point either.
    with pytest.raises(PlanDrift):
        resumed.resume((4.0, 50.0))


def test_resumed_run_matches_uninterrupted_plan_and_totals():
    feed = [(0.0, 0.0), (4.0, 0.0), (10.0, 0.0), (10.0, 5.0), (10.0, 10.0)]
    samples = [{'x': 2.0, 'y': 0.0}, {'x': 10.0, 'y': 3.0}, {'x': 10.0, 'y': 8.0}]

    # Uninterrupted reference run, samples arriving before completion.
    route_ref = PlannedRoute(ROUTE)
    reference = ResumableMission.start('ref', route_ref, feed[0])
    ref_distance = reference.update(feed[1])
    reference.collect(samples[0])
    ref_distance += reference.update(feed[2])
    ref_distance += reference.update(feed[3])
    reference.collect(samples[1])
    reference.collect(samples[2])
    ref_distance += reference.update(feed[4])

    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m6', route, feed[0])
    total = mission.update(feed[1])
    mission.collect(samples[0])
    mission.fault('fault')
    resumed = ResumableMission.restore(PlannedRoute(ROUTE), mission.checkpoint())
    resumed.rejoin((0.0, 0.0))                          # towed back to origin
    resumed.resume(feed[1])
    total += resumed.update(feed[2])
    assert resumed.collect(samples[1])                 # new after fault
    total += resumed.update(feed[3])
    assert resumed.collect(samples[2])
    assert resumed.collect(samples[0]) is False        # pre-fault redelivery, counted once
    total += resumed.update(feed[4])

    assert resumed.state == reference.state == 'completed'
    assert total == pytest.approx(ref_distance)
    assert resumed.covered_distance == pytest.approx(reference.covered_distance)
    assert resumed.sample_count == reference.sample_count == 3
    # The resumed trajectory sits on the same planned geometry.
    for arc in (4.0, 10.0, 15.0, 20.0):
        assert resumed.route.point_at(arc) == reference.route.point_at(arc)
    assert resumed.route.digest == reference.route.digest


def test_backtracking_over_covered_ground_adds_zero():
    route = PlannedRoute(ROUTE)
    mission = ResumableMission.start('m7', route, (0.0, 0.0))
    assert mission.update((6.0, 0.0)) == pytest.approx(6.0)
    assert mission.update((3.0, 0.0)) == pytest.approx(0.0)   # drove back
    assert mission.covered_distance == pytest.approx(6.0)
    assert mission.update((9.0, 0.0)) == pytest.approx(3.0)   # only new ground


def test_running_off_plan_is_rejected_but_event_history_survives():
    route = PlannedRoute(ROUTE, cross_track_tolerance=0.5)
    mission = ResumableMission.start('m8', route, (0.0, 0.0))
    with pytest.raises(PlanDrift):
        mission.update((3.0, 5.0))
    mission.update((2.0, 0.0))
    mission.fault('fault')
    kinds = [e['event'] for e in mission.events]
    restored = ResumableMission.restore(PlannedRoute(ROUTE, cross_track_tolerance=0.5),
                                        mission.checkpoint())
    assert [e['event'] for e in restored.events] == kinds
    assert kinds == ['start', 'fault']


def test_ledger_roundtrip_and_progress_ledger_semantics():
    ledger = ProgressLedger()
    ledger.record_position(5.0)
    assert ledger.record_position(2.0) is False
    route = PlannedRoute(ROUTE)
    assert ledger.add_sample({'x': 1.0, 'y': 0.0}, route=route)
    assert ledger.add_sample({'x': 1.004, 'y': 0.0}, route=route) is False  # 1 cm grid
    restored = ProgressLedger.from_dict(ledger.to_dict())
    assert restored.furthest_arc == 5.0
    assert restored.samples == ledger.samples
