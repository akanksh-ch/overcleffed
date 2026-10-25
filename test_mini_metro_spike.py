"""Regression tests for the isolated Mini Metro experiment (no inference calls)."""
import copy
import os
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
UPSTREAM = Path(os.environ.get("MINI_METRO_UPSTREAM", "/tmp/overcleffed-mini-metro-upstream"))
if not (UPSTREAM / "src/env.py").exists():
    raise unittest.SkipTest("set MINI_METRO_UPSTREAM to the pinned external simulator")
sys.path.insert(0, str(UPSTREAM / "src"))
try:
    from env import MiniMetroEnv
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"isolated simulator dependencies unavailable: {error}") from error

from mini_metro_spike import (Adapter, Decisions, Recording, build_tree, heuristic, make_env,
                              parse_choice, quality, run_episode, tree_clef)


class SpikeTests(unittest.TestCase):
    def setUp(self):
        self.env = MiniMetroEnv(dt_ms=16)
        self.env.reset(seed=42)
        self.adapter = Adapter(self.env)

    def test_menu_is_legal_and_enumeration_does_not_change_world(self):
        before = self.env.observe()
        snap = self.adapter.snapshot()
        self.assertEqual(before['structured'], self.env.observe()['structured'])
        self.assertEqual(len(snap['menu']), 4)  # noop plus three unordered pairs
        for name in snap['menu']:
            env = MiniMetroEnv(dt_ms=16)
            env.reset(seed=42)
            adapter = Adapter(env)
            candidate = adapter.snapshot()['menu'][name]
            _, _, _, info = adapter.tick(candidate)
            self.assertTrue(info['action_ok'], name)
        self.assertIn('create S0-S1', snap['menu'])

    def test_main_thread_only_mutation_and_snapshot_is_detached(self):
        snap = self.adapter.snapshot()
        snap['state']['stations'].clear()
        errors = []
        def worker():
            try:
                self.adapter.tick({'type': 'noop'})
            except RuntimeError as error:
                errors.append(str(error))
        t = threading.Thread(target=worker)
        t.start(); t.join(timeout=1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(self.env.mediator.time_ms, 0)
        self.assertEqual(len(self.env.mediator.stations), 3)

    def test_route_and_fleet_candidates_execute_on_the_real_simulator(self):
        self.adapter.tick(self.adapter.snapshot()['menu']['create S0-S1'])
        self.adapter.tick(self.adapter.snapshot()['menu']['assign locomotive L0'])
        for name in self.adapter.snapshot()['menu']:
            env = MiniMetroEnv(dt_ms=16)
            env.reset(seed=42)
            adapter = Adapter(env)
            adapter.tick(adapter.snapshot()['menu']['create S0-S1'])
            adapter.tick(adapter.snapshot()['menu']['assign locomotive L0'])
            _, _, _, info = adapter.tick(adapter.snapshot()['menu'][name])
            self.assertTrue(info['action_ok'], name)

    def test_recording_clears_background_before_each_frame(self):
        recording = Recording('test')
        recording(self.adapter, [])
        self.assertEqual(tuple(recording.last.get_at((5, 300)))[:3], (247, 245, 239))
        recording.surface.fill((255, 0, 0))
        recording(self.adapter, [])
        self.assertEqual(tuple(recording.last.get_at((5, 300)))[:3], (247, 245, 239))

    @unittest.skipUnless(shutil.which('ffprobe') and os.path.exists('/usr/bin/ffmpeg'), 'ffmpeg/ffprobe missing')
    def test_video_is_30fps_and_recording_does_not_change_sim_or_decisions(self):
        def episode(video):
            env = MiniMetroEnv(dt_ms=16)
            env.reset(seed=42)
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
                time.sleep(0.0001)
            rec = Recording('t', video) if video else None
            summary, events, telemetry = run_episode(
                Adapter(env), heuristic, sim_cap_ms=6000, wall_cap_s=15,
                clock=lambda: clock[0], sleep=sleep, record=rec)
            if rec:
                rec.close()
            keys = ('status', 'snapshot_sim_ms', 'apply_sim_ms', 'choice', 'simulated_age_ms')
            return (env.mediator.time_ms, env.mediator.steps, env.mediator.deliveries,
                    [{k: e.get(k) for k in keys} for e in events], telemetry)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'out.mp4'
            self.assertEqual(episode(path), episode(None))
            info = subprocess.check_output(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                 '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames',
                 '-of', 'default=nw=1', str(path)], text=True)
            facts = dict(line.split('=') for line in info.split())
            self.assertEqual((facts['codec_name'], facts['pix_fmt'], facts['width'], facts['height'], facts['r_frame_rate']),
                             ('h264', 'yuv420p', '960', '540', '30/1'))
            self.assertAlmostEqual(int(facts['nb_read_frames']), 6 * 30 + 1, delta=2)  # 6 sim s at 30 fps
            self.assertTrue(path.with_suffix('.png').exists())

    def test_spawn_pressure_changes_demand_and_stays_seed_deterministic(self):
        def spawned(steps):
            env = make_env(42, steps)
            for _ in range(1875):  # 30 simulated seconds of noop
                env.step(None)
            return len(env.mediator.passengers)
        base, dense = spawned(900), spawned(300)
        self.assertEqual(spawned(300), dense)
        self.assertGreaterEqual(dense, base + 6)
        self.assertEqual(make_env(42, 300).mediator.passenger_spawning_interval_step, 300)

    def test_unserved_stations_counts_stations_off_every_line(self):
        from mini_metro_spike import metrics
        self.assertEqual(metrics(self.adapter)['unserved'], 3)
        self.adapter.tick(self.adapter.snapshot()['menu']['create S0-S1'])
        self.assertEqual(metrics(self.adapter)['unserved'], 1)

    def test_quality_summarises_hand_checked_telemetry(self):
        rows = [dict(sim_ms=0, deliveries=0, waiting=2, unserved=3, oldest_wait_ms=0, overdue=0),
                dict(sim_ms=1000, deliveries=0, waiting=4, unserved=1, oldest_wait_ms=3000, overdue=0),
                dict(sim_ms=2000, deliveries=2, waiting=6, unserved=0, oldest_wait_ms=9000, overdue=1)]
        got = quality(dict(sim_ms=2000, deliveries=2, game_over=True), rows)
        self.assertEqual(got, dict(survival_s=2.0, game_over=True, deliveries=2,
                                   deliveries_per_min=60.0, first_delivery_s=2.0,
                                   mean_waiting=4.0, mean_unserved=4 / 3,
                                   peak_oldest_wait_s=9.0, peak_overdue=1))

    def test_reset_prevents_old_adapter_from_touching_new_episode(self):
        self.env.reset(seed=42)
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            self.adapter.tick({'type': 'noop'})
        self.assertEqual(self.env.mediator.time_ms, 0)

    def test_response_identity_deadline_and_unknown_choice_rejected(self):
        snap = self.adapter.snapshot()
        good = {'episode_id': snap['episode_id'], 'snapshot_id': snap['snapshot_id'],
                'choice': 'create S0-S1'}
        for changes, age, reason in [
            ({'episode_id': 'retired'}, 0.1, 'episode_mismatch'),
            ({'snapshot_id': 999}, 0.1, 'snapshot_mismatch'),
            ({}, 3.0, 'late'),
            ({'choice': 'invented'}, 0.1, 'unknown_choice'),
        ]:
            action, status = self.adapter.resolve(snap, dict(good, **changes), age)
            self.assertIsNone(action)
            self.assertEqual(status, reason)
        action, status = self.adapter.resolve(snap, good, 2.99)
        self.assertEqual(status, 'accepted')
        self.assertEqual(action['type'], 'create_path')
        self.assertEqual(self.env.mediator.time_ms, 0)

    def test_new_episode_and_changed_route_invalidate_old_snapshot(self):
        snap = self.adapter.snapshot()
        response = {'episode_id': snap['episode_id'], 'snapshot_id': snap['snapshot_id'],
                    'choice': 'create S0-S1'}
        replacement = Adapter(self.env)
        self.assertEqual(replacement.resolve(snap, response, 0.1)[1], 'episode_mismatch')
        self.adapter.tick(snap['menu']['create S0-S1'])
        route_snap = self.adapter.snapshot()
        extension = next(k for k in route_snap['menu'] if k.startswith('extend'))
        self.env.step({'type': 'remove_path', 'path_index': 0}, dt_ms=0)
        response = dict(response, snapshot_id=route_snap['snapshot_id'], choice=extension)
        self.assertEqual(self.adapter.resolve(route_snap, response, 0.1)[1], 'no_longer_legal')

    def test_fleet_menu_revalidates_exhausted_resources(self):
        self.adapter.tick(self.adapter.snapshot()['menu']['create S0-S1'])
        snap = self.adapter.snapshot()
        key = 'assign locomotive L0'
        self.assertIn(key, snap['menu'])
        while self.env.mediator.available_locomotives:
            self.env.mediator.assign_locomotive(self.env.mediator.paths[0])
        response = {'episode_id': snap['episode_id'], 'snapshot_id': snap['snapshot_id'], 'choice': key}
        self.assertEqual(self.adapter.resolve(snap, response, 0.2)[1], 'no_longer_legal')

    def test_delayed_worker_never_blocks_ticks_or_starts_second_request(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(snapshot):
            entered.set()
            release.wait(1)
            return 'create S0-S1'
        clock = [0.0]
        decisions = Decisions(self.adapter, delayed, clock=lambda: clock[0])
        self.assertTrue(decisions.request())
        self.assertTrue(entered.wait(1))
        for _ in range(10):
            self.adapter.tick(None)
        self.assertEqual(self.env.mediator.time_ms, 160)
        clock[0] = 3.01
        self.assertIsNone(decisions.poll())
        self.assertEqual(decisions.events[-1]['status'], 'deadline_expired')
        self.assertFalse(decisions.request())
        release.set()
        decisions.thread.join(timeout=1)
        self.assertIsNone(decisions.poll())
        self.assertEqual(decisions.events[-1]['status'], 'late')
        self.assertEqual(len(self.env.mediator.paths), 0)

    def test_timely_worker_returns_action_only_for_main_tick(self):
        decisions = Decisions(self.adapter, lambda snap: 'create S0-S1')
        decisions.request()
        decisions.thread.join(timeout=1)
        action = decisions.poll()
        self.assertEqual(action['type'], 'create_path')
        self.assertEqual(len(self.env.mediator.paths), 0)
        self.adapter.tick(action)
        self.assertEqual(len(self.env.mediator.paths), 1)
        self.assertEqual(decisions.events[-1]['status'], 'accepted')

    def test_policy_error_is_logged_not_heuristic_fallback(self):
        def broken(snapshot):
            raise ValueError('fake HTTP failure')
        decisions = Decisions(self.adapter, broken)
        decisions.request(); decisions.thread.join(timeout=1)
        self.assertIsNone(decisions.poll())
        self.assertEqual(decisions.events[-1]['status'], 'error')
        self.assertIn('fake HTTP failure', decisions.events[-1]['error'])
        self.assertEqual(len(self.env.mediator.paths), 0)

    def test_parser_rejects_invalid_probabilities_instead_of_fallback(self):
        payload = {'answers': {'route_operation': {'probabilities': {'noop': 0.2, 'create S0-S1': 0.8}}}}
        self.assertEqual(parse_choice(payload, {'noop': {}, 'create S0-S1': {}}), 'create S0-S1')
        for probabilities in ({'fake': 1.0}, {'noop': float('nan')}, {'noop': -1.0}, {}, {'noop': 0.0}):
            bad = copy.deepcopy(payload)
            bad['answers']['route_operation']['probabilities'] = probabilities
            with self.assertRaises(ValueError):
                parse_choice(bad, {'noop': {}})

    def test_runner_wall_cap_does_not_wait_for_worker_or_apply_pending_action(self):
        release = threading.Event()
        entered = threading.Event()
        def delayed(snapshot):
            entered.set()
            release.wait(1)
            return 'create S0-S1'
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
            entered.wait(0.01)
        try:
            summary, events, _ = run_episode(
                self.adapter, delayed, sim_cap_ms=12000, wall_cap_s=0.1,
                clock=lambda: clock[0], sleep=sleep)
            self.assertEqual(summary['ending'], 'wall_cap')
            self.assertLessEqual(summary['wall_s'], 0.1)
            self.assertEqual(len(self.env.mediator.paths), 0)
            self.assertEqual(events[-1]['status'], 'episode_ended_pending')
        finally:
            release.set()

    def test_runner_has_fixed_ticks_continuous_opportunities_and_call_cap(self):
        # Removing pacing, cadence, or per-run call limiting must fail this check.
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
            time.sleep(0.0001)  # yield to the real worker, not simulated inference
        summary, events, telemetry = run_episode(
            self.adapter, heuristic, sim_cap_ms=12000, wall_cap_s=15,
            max_calls=2, clock=lambda: clock[0], sleep=sleep)
        self.assertEqual(summary['sim_ms'], 12000)
        self.assertGreaterEqual(summary['wall_s'], 12.0)
        self.assertEqual(summary['calls'], 2)
        self.assertEqual([e['snapshot_sim_ms'] for e in events if e['status'] == 'requested'], [0, 5008])
        self.assertEqual(self.env.mediator.steps, 750)
        self.assertTrue(any(e['status'] == 'call_cap' for e in events))
        self.assertEqual(summary['ending'], 'sim_cap')
        self.assertGreater(len(telemetry), 1)


class TreeTests(unittest.TestCase):
    def setUp(self):
        env = MiniMetroEnv(dt_ms=16)
        env.reset(seed=42)
        self.adapter = Adapter(env)

    def snap(self, *ops):
        for op in ops:
            self.adapter.tick(self.adapter.snapshot()['menu'][op])
        for _ in range(1000):  # 16 s: every station has spawned passengers
            self.adapter.tick(None)
        return self.adapter.snapshot()

    def test_tree_groups_legal_leaves_by_station_and_fix_and_prunes_churn(self):
        snap = self.snap()
        tree = build_tree(snap['menu'], snap['state'])
        self.assertEqual(set(tree), {'S0', 'S1', 'S2'})
        self.assertEqual(tree['S0']['connect'], ['create S0-S1', 'create S0-S2'])
        self.assertEqual(tree['S0']['wait'], ['noop'])
        snap = self.snap('create S0-S1', 'assign locomotive L0')
        tree = build_tree(snap['menu'], snap['state'])
        self.assertEqual(tree['S2']['extend'], ['extend L0 start to S2', 'extend L0 end to S2'])
        self.assertEqual(tree['S0'].get('add_train'), ['assign locomotive L0'])
        self.assertNotIn('unassign locomotive L0', sum((v for t in tree.values() for v in t.values()), []))
        self.assertNotIn('detach carriage L0', sum((v for t in tree.values() for v in t.values()), []))

    def test_tree_drops_capacity_fixes_for_station_with_nobody_waiting(self):
        snap = self.adapter.snapshot()
        snap['state']['stations'][0]['waiting'] = 0
        tree = build_tree(snap['menu'], snap['state'])
        self.assertNotIn('add_train', tree['S0'])

    def test_walk_picks_urgent_station_then_fix_then_leaf_in_one_request(self):
        snap = self.snap('create S0-S1', 'assign locomotive L0')
        calls = []
        def ask(state, questions, **kw):
            calls.append(questions)
            return {'answers': {
                'urgency_S0': {'noul': 0.1}, 'urgency_S1': {'noul': 0.2}, 'urgency_S2': {'noul': 0.9},
                'fix_S2': {'probabilities': {'extend': 0.7, 'wait': 0.3}},
                'fix_S0': {'probabilities': {'wait': 1.0}},  # other stations' answers are ignored
                'leaf_S2_extend': {'probabilities': {'extend L0 end to S2': 0.8, 'extend L0 start to S2': 0.2}}}}
        got = tree_clef(snap, ask=ask)
        self.assertEqual(got['choice'], 'extend L0 end to S2')
        self.assertEqual(got['trace'], ['S2', 'extend', 'extend L0 end to S2'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {'urgency_S0', 'urgency_S1', 'urgency_S2', 'fix_S0', 'fix_S1', 'fix_S2',
                                         'leaf_S2_extend'})  # single-leaf fix types get no leaf question
        self.assertEqual(calls[0]['fix_S2']['criteria'].keys(), {'wait', 'extend'})

    def test_walk_makes_no_request_when_only_wait_remains(self):
        snap = self.adapter.snapshot()
        snap['menu'] = {'noop': {'type': 'noop'}}
        got = tree_clef(snap, ask=lambda *a: self.fail('no request expected'))
        self.assertEqual(got['choice'], 'noop')

    def test_walk_rejects_malformed_answers_without_fallback(self):
        snap = self.snap()
        for bad in ({}, {'answers': {}}, {'answers': {'urgency_S0': {'noul': float('nan')}, 'urgency_S1': {'noul': 0.1}, 'urgency_S2': {'noul': 0.1}}},
                    {'answers': {'urgency_S0': {'noul': 0.0}, 'urgency_S1': {'noul': 0.0}, 'urgency_S2': {'noul': 0.0}}}):
            with self.assertRaises(ValueError):
                tree_clef(snap, ask=lambda state, questions, bad=bad, **kw: bad)
        def wrong_fix(state, questions, **kw):
            return {'answers': {'urgency_S0': {'noul': 1.0}, 'urgency_S1': {'noul': 0.0}, 'urgency_S2': {'noul': 0.0},
                                'fix_S0': {'probabilities': {'invented': 1.0}}}}
        with self.assertRaises(ValueError):
            tree_clef(snap, ask=wrong_fix)


if __name__ == '__main__':
    unittest.main()
