"""Bounded, assisted-control Mini Metro comparison. No training or kitchen imports.

Run with the isolated Python environment; see docs/mini-metro-spike.md.
The pinned external simulator is only imported by main(), never vendored here.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen
import uuid

UPSTREAM_COMMIT = 'ef1be2dfdc6e1dadafa9bf2c41d1ce5f081adc4d'
TICK_MS = 16
DECISION_MS = 5000
DEADLINE_S = 3.0


class Adapter:
    """Main-thread-only environment boundary; enumeration contains no strategy."""

    def __init__(self, env):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('construct adapter on main thread')
        if env.mediator.map_definition.map_id != 'classic':
            raise ValueError('this bounded spike supports only Classic')
        self.env = env
        self.mediator = env.mediator  # reset invalidates every outstanding snapshot
        self.episode_id = uuid.uuid4().hex
        self.sequence = 0

    def check_owner(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('environment access is main-thread-only')
        if self.env.mediator is not self.mediator:
            raise RuntimeError('environment reset: construct a new episode adapter')

    def menu(self):
        self.check_owner()
        m = self.mediator
        if m.is_game_over:
            return {}
        stations = m.stations
        menu = {'noop': {'type': 'noop'}}
        # All unordered pairs, no distance/destination ranking or top-k filtering.
        if len(m.paths) < m.unlocked_num_paths:
            for i, j in itertools.combinations(range(len(stations)), 2):
                menu[f'create S{i}-S{j}'] = {
                    'type': 'create_path', 'station_ids': [stations[i].id, stations[j].id]}
        for k, path in enumerate(m.paths):
            ids = [s.id for s in path.stations]
            if not path.is_looped:
                for i, station in enumerate(stations):
                    if station.id not in ids:
                        for side, route in (('start', [station.id] + ids), ('end', ids + [station.id])):
                            menu[f'extend L{k} {side} to S{i}'] = {
                                'type': 'replace_path', 'path_id': path.id, 'station_ids': route}
            checks = [
                ('assign locomotive', 'assign_locomotive', m.can_assign_locomotive),
                ('unassign locomotive', 'unassign_locomotive', m.can_queue_locomotive_unassignment),
                ('attach carriage', 'attach_carriage', m.can_attach_carriage),
                ('detach carriage', 'detach_carriage', m.can_detach_carriage),
            ]
            for label, kind, legal in checks:
                if legal(path):
                    menu[f'{label} L{k}'] = {'type': kind, 'path_id': path.id}
        return menu

    def state(self):
        self.check_owner()
        m = self.mediator
        station_labels = {s.id: f'S{i}' for i, s in enumerate(m.stations)}
        return {
            'sim_ms': m.time_ms, 'deliveries': m.deliveries,
            'max_wait_ms': m.passenger_max_wait_time_ms,
            'overdue_game_over_threshold': m.overdue_passenger_threshold,
            'available_locomotives': m.available_locomotives,
            'available_carriages': m.available_carriages,
            'stations': [dict(label=f'S{i}', shape=s.shape.type.name,
                              position=[s.position.left, s.position.top],
                              waiting=len(s.passengers),
                              oldest_wait_ms=max((p.wait_ms for p in s.passengers), default=0),
                              destinations={shape: sum(p.destination_shape.type.name == shape for p in s.passengers)
                                            for shape in sorted({p.destination_shape.type.name for p in s.passengers})})
                         for i, s in enumerate(m.stations)],
            'lines': [dict(label=f'L{i}', stations=[station_labels[s.id] for s in path.stations],
                           locomotives=sum(t.path_id == path.id for t in m.metros),
                           queued=sum(t.path_id == path.id and t.is_unassignment_queued for t in m.metros),
                           onboard=sum(len(t.passengers) for t in m.metros if t.path_id == path.id),
                           capacity=sum(t.capacity for t in m.metros if t.path_id == path.id))
                      for i, path in enumerate(m.paths)],
        }

    def snapshot(self):
        self.check_owner()
        self.sequence += 1
        return {'episode_id': self.episode_id, 'snapshot_id': self.sequence,
                'sim_ms': self.mediator.time_ms, 'state': self.state(), 'menu': self.menu()}

    def resolve(self, snapshot, response, age_s):
        """Validate identity, age and *current* legality, without mutating the world."""
        self.check_owner()
        if snapshot['episode_id'] != self.episode_id or response.get('episode_id') != self.episode_id:
            return None, 'episode_mismatch'
        if (response.get('snapshot_id') != snapshot['snapshot_id']
                or snapshot['snapshot_id'] != self.sequence):
            return None, 'snapshot_mismatch'
        if age_s >= DEADLINE_S or age_s < 0:
            return None, 'late'
        key = response.get('choice')
        if not isinstance(key, str) or key not in snapshot['menu']:
            return None, 'unknown_choice'
        action = snapshot['menu'][key]
        if self.menu().get(key) != action:
            return None, 'no_longer_legal'
        return copy.deepcopy(action), 'accepted'

    def tick(self, action):
        self.check_owner()
        if action is None:
            action = {'type': 'noop'}
        if action not in self.menu().values():
            raise ValueError('action is not currently legal')
        command = copy.deepcopy(action)
        if 'station_ids' in command:
            indices = {s.id: i for i, s in enumerate(self.mediator.stations)}
            command['stations'] = [indices[s] for s in command.pop('station_ids')]
        result = self.env.step(command, dt_ms=TICK_MS)
        if not result[3]['action_ok']:
            # Upstream refuses to advance on a rejected command. Advance the SAME
            # physics tick by noop, without choosing any replacement strategy.
            obs, reward, done, _ = self.env.step(None, dt_ms=TICK_MS)
            return obs, reward + result[1], done, {'action_ok': False}
        return result


class Decisions:
    """One daemon inference worker; only detached snapshots cross the boundary."""

    def __init__(self, adapter, choose, *, clock=time.monotonic):
        self.adapter, self.choose, self.clock = adapter, choose, clock
        self.pending = None
        self.thread = None
        self.results = queue.Queue()
        self.events = []
        self.calls = 0

    def request(self):
        self.adapter.check_owner()
        if self.pending is not None:
            return False
        snap = self.adapter.snapshot()
        start = self.clock()
        self.pending = {'snapshot': snap, 'start': start, 'expired': False}
        self.calls += 1
        self.events.append(dict(status='requested', snapshot_id=snap['snapshot_id'],
                                snapshot_sim_ms=snap['sim_ms'], state=snap['state'], menu=snap['menu']))

        choose, results, clock = self.choose, self.results, self.clock
        worker_snapshot = copy.deepcopy(snap)

        def work():
            response = {'episode_id': worker_snapshot['episode_id'],
                        'snapshot_id': worker_snapshot['snapshot_id']}
            try:
                chosen = choose(worker_snapshot)
                response.update(chosen if isinstance(chosen, dict) else {'choice': chosen})
            except Exception as error:
                response['error'] = f'{type(error).__name__}: {error}'[:500]
            results.put((response, clock()))

        self.thread = threading.Thread(target=work, name='metro-inference', daemon=True)
        self.thread.start()
        return True

    def poll(self):
        self.adapter.check_owner()
        if self.pending is None:
            return None
        pending = self.pending
        snap = pending['snapshot']
        now = self.clock()
        try:
            response, finished = self.results.get_nowait()
        except queue.Empty:
            if now - pending['start'] >= DEADLINE_S and not pending['expired']:
                pending['expired'] = True
                self.events.append(dict(status='deadline_expired', snapshot_id=snap['snapshot_id'],
                                        snapshot_sim_ms=snap['sim_ms'], age_s=now - pending['start']))
            return None
        self.pending = None
        age = now - pending['start']
        action, status = self.adapter.resolve(snap, response, age)
        if 'error' in response and status != 'late':
            action, status = None, 'error'
        if pending['expired']:
            action, status = None, 'late'
        self.events.append({**response, 'status': status, 'snapshot_id': snap['snapshot_id'],
                            'snapshot_sim_ms': snap['sim_ms'], 'apply_sim_ms': self.adapter.mediator.time_ms,
                            'latency_s': finished - pending['start'], 'age_s': age,
                            'simulated_age_ms': self.adapter.mediator.time_ms - snap['sim_ms']})
        return action


def parse_choice(payload, menu):
    try:
        probabilities = payload['answers']['route_operation']['probabilities']
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError('empty probabilities')
        for name, probability in probabilities.items():
            if (name not in menu or type(probability) not in (int, float)
                    or not math.isfinite(probability) or probability < 0):
                raise ValueError('invalid choice or probability')
        if sum(probabilities.values()) <= 0:
            raise ValueError('zero probability mass')
        return max(probabilities, key=probabilities.get)
    except (KeyError, TypeError) as error:
        raise ValueError('malformed Clef answer') from error


def clef(snapshot):
    """Existing Clef protocol. This is not a distinct Jev integration."""
    instructions = (
        'Manage Mini Metro continuously: deliver passengers before station waiting reaches 40000 ms. '
        'Passengers need a station of their destination shape, possibly via transfers. '
        'New routes have NO train: assign a locomotive in a subsequent decision to serve them. '
        'Extend endpoints to connect stations; allocate trains/carriages to congested lines. '
        'One atomic operation now; another decision in 5 simulated seconds. '
        'Choose only from the legal menu. Noop is permitted when no change is useful.')
    body = {'model': 'clef-flash', 'state': json.dumps(snapshot['state']),
            'questions': {'route_operation': {'type': 'choice', 'instructions': instructions,
                                              'criteria': {name: name for name in snapshot['menu']}}}}
    request = Request('http://127.0.0.1:8080/v1/systemone',
                      data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    # Finite socket timeout; the controller independently rejects at 3 seconds.
    with urlopen(request, timeout=2.5) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('oversized Clef response')
    payload = json.loads(raw)
    return {'choice': parse_choice(payload, snapshot['menu']),
            'probabilities': payload['answers']['route_operation']['probabilities']}


def heuristic(snapshot):
    """Explicit baseline, NOT part of legal-action generation or model control."""
    state, menu = snapshot['state'], snapshot['menu']
    stations = {s['label']: s for s in state['stations']}
    lines = state['lines']
    for line in lines:
        key = f"assign locomotive {line['label']}"
        if line['locomotives'] == 0 and key in menu:
            return key
    served = {s for line in lines for s in line['stations']}
    options = []
    for key in menu:
        if key.startswith('extend'):
            _, label, side, _, target = key.split()
            if target not in served:
                line = next(line for line in lines if line['label'] == label)
                endpoint = line['stations'][0 if side == 'start' else -1]
                distance = math.dist(stations[target]['position'], stations[endpoint]['position'])
                options.append((distance, key))
    if options:
        return min(options)[1]
    if not lines:
        pairs = []
        for key in menu:
            if key.startswith('create'):
                a, b = key.split()[1].split('-')
                pairs.append((stations[a]['shape'] == stations[b]['shape'],
                              math.dist(stations[a]['position'], stations[b]['position']), key))
        if pairs:
            return min(pairs)[2]
    # Spare capacity goes to the line with most waiting passengers; keep at
    # least three waiting before adding a train, then use carriages if needed.
    pressure = sorted(((sum(stations[s]['waiting'] for s in line['stations']), line['label'])
                       for line in lines), reverse=True)
    for waiting, label in pressure:
        if waiting >= 3:
            for kind in ('assign locomotive', 'attach carriage'):
                if f'{kind} {label}' in menu:
                    return f'{kind} {label}'
    return 'noop'


def metrics(adapter):
    adapter.check_owner()
    m = adapter.mediator
    waiting = [p for s in m.stations for p in s.passengers]
    return dict(sim_ms=m.time_ms, deliveries=m.deliveries, stations=len(m.stations),
                waiting=len(waiting), onboard=sum(len(t.passengers) for t in m.metros),
                oldest_wait_ms=max((p.wait_ms for p in waiting), default=0),
                overdue=sum(p.wait_ms >= m.passenger_max_wait_time_ms for p in waiting),
                game_over=m.is_game_over)


def run_episode(adapter, choose, *, sim_cap_ms=120000, wall_cap_s=150, max_calls=24,
                clock=time.monotonic, sleep=time.sleep, record=None):
    decisions = Decisions(adapter, choose, clock=clock)
    start = clock()
    next_decision = 0
    next_sample = 0
    next_frame = 0
    telemetry = []
    total_reward = 0
    max_waiting = max_oldest_wait_ms = 0
    action_rejections = 0
    ending = 'sim_cap'
    while adapter.mediator.time_ms < sim_cap_ms:
        if adapter.mediator.is_game_over:
            ending = 'game_over'
            break
        # Pace the next physics tick to real time. Catch up with fixed ticks if
        # local rendering/HTTP scheduling falls behind; never enlarge dt_ms.
        target = start + (adapter.mediator.time_ms + TICK_MS) / 1000
        remaining = target - clock()
        if remaining > 0:
            sleep(min(remaining, max(0, start + wall_cap_s - clock())))
        if clock() - start >= wall_cap_s:
            ending = 'wall_cap'
            break
        action = decisions.poll()
        if adapter.mediator.time_ms >= next_decision:
            next_decision += DECISION_MS
            if decisions.calls >= max_calls:
                decisions.events.append(dict(status='call_cap', sim_ms=adapter.mediator.time_ms))
            elif not decisions.request():
                decisions.events.append(dict(status='busy', sim_ms=adapter.mediator.time_ms))
        _, reward, _, info = adapter.tick(action)
        total_reward += reward
        if not info['action_ok']:
            action_rejections += 1
            decisions.events.append(dict(status='upstream_rejected', sim_ms=adapter.mediator.time_ms,
                                        action=action))
        current = metrics(adapter)
        max_waiting = max(max_waiting, current['waiting'])
        max_oldest_wait_ms = max(max_oldest_wait_ms, current['oldest_wait_ms'])
        if adapter.mediator.time_ms >= next_sample:
            telemetry.append(current)
            next_sample += 1000
        if record is not None and adapter.mediator.time_ms >= next_frame:
            record(adapter, decisions.events)
            next_frame += 500
    # Do not wait on inference or mutate a finished episode. The worker has no
    # environment reference; any result arriving after this point is discarded.
    if decisions.pending is not None:
        snap = decisions.pending['snapshot']
        decisions.events.append(dict(status='episode_ended_pending', snapshot_id=snap['snapshot_id'],
                                    snapshot_sim_ms=snap['sim_ms'], age_s=clock() - decisions.pending['start']))
    final = metrics(adapter)
    if final['game_over']:
        ending = 'game_over'
    telemetry.append(final)
    if record is not None:
        record(adapter, decisions.events)
    latencies = [e['latency_s'] for e in decisions.events if 'latency_s' in e]
    summary = dict(final, ending=ending, wall_s=clock() - start, calls=decisions.calls,
                   reward=total_reward, upstream_rejections=action_rejections,
                   max_waiting=max_waiting, max_oldest_wait_ms=max_oldest_wait_ms,
                   statuses={status: sum(e['status'] == status for e in decisions.events)
                             for status in sorted({e['status'] for e in decisions.events})},
                   latency_s=latencies)
    return summary, decisions.events, telemetry


class Recording:
    """Small, privileged-state GIF, not a pixel-input policy recording."""

    def __init__(self, label):
        import pygame
        from rendering.game_renderer import GameRenderer
        self.label = label
        self.renderer = GameRenderer()
        self.surface = pygame.Surface((1920, 1080))
        self.frames = []

    def __call__(self, adapter, events):
        import pygame
        from PIL import Image, ImageDraw
        from config import screen_color
        self.surface.fill(screen_color)
        self.renderer.draw(self.surface, adapter.mediator, alpha=1.0)
        image = Image.frombytes('RGB', self.surface.get_size(), pygame.image.tobytes(self.surface, 'RGB'))
        image = image.resize((768, 432))
        draw = ImageDraw.Draw(image)
        row = metrics(adapter)
        status = next((e.get('choice', e['status']) for e in reversed(events)
                       if e['status'] != 'requested'), 'waiting')
        draw.rectangle((0, 0, 768, 46), fill='white')
        draw.text((8, 4), f"{self.label} | ASSISTED STRUCTURED CONTROL | {row['sim_ms']/1000:.1f}s | delivered {row['deliveries']} | waiting {row['waiting']}", fill='black')
        draw.text((8, 25), str(status)[:100], fill='black')
        self.frames.append(image.quantize(colors=128))

    def save(self, path):
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=500, loop=0)
        self.frames[-1].save(path.with_suffix('.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=Path('/tmp/overcleffed-mini-metro-upstream'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], timeout=5, text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.upstream), 'status', '--porcelain', '--untracked-files=no'], timeout=5, text=True)
    if commit != UPSTREAM_COMMIT or dirty:
        parser.error('upstream must be clean and at the recorded commit')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    sys.path.insert(0, str(args.upstream.resolve() / 'src'))
    from env import MiniMetroEnv
    manifest = dict(upstream_commit=commit, seed=42, tick_ms=TICK_MS,
                    decision_ms=DECISION_MS, deadline_s=DEADLINE_S, http_socket_timeout_s=2.5,
                    sim_cap_ms=120000, wall_cap_s=150, max_calls=24, python=sys.version,
                    assistance='structured state and restricted atomic route/fleet operations',
                    model='clef-flash', endpoint='http://127.0.0.1:8080/v1/systemone')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    summaries = {}
    for label, choose in [('heuristic', heuristic), ('clef', clef)]:
        env = MiniMetroEnv(dt_ms=TICK_MS)
        env.reset(seed=42)
        recording = Recording(label)
        print(f'Starting {label}: seed 42, fixed 16ms, 120s sim / 150s wall cap', flush=True)
        summary, events, telemetry = run_episode(Adapter(env), choose, record=recording)
        summaries[label] = summary
        for name, data in [('summary', summary), ('events', events), ('telemetry', telemetry)]:
            (args.output / f'{label}-{name}.json').write_text(json.dumps(data, indent=2))
        export_start = time.monotonic()
        recording.save(args.output / f'{label}.gif')
        print(json.dumps(dict(policy=label, **summary, export_wall_s=time.monotonic()-export_start)), flush=True)
    (args.output / 'comparison.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
