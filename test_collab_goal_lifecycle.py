"""Standard-library lifecycle regressions; no Collab fork or model required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'collab_bench'))
from gemma_planner import GemmaPlanner, sentence


def goal(action, item=None, station=None):
    return dict(action=action, item=item, station=station)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        mdp = NS(utensil_list=['pot0'], recipes={'pot': {'boiled_egg': {'recipe': ['egg']}}},
                 need_dish={}, get_counter_locations=lambda: [(2, 1)],
                 utensil_state_dict={'pot0': {'position': (3, 1)}}, utensils={'pot': {'operation': 'cook'}})
        with patch('gemma_planner.recipe_text', return_value='boil egg'):
            self.p = GemmaPlanner(mdp, 'boiled_egg')
        self.ut = {'pot0': dict(status='empty', contents=[], product=None)}
        self.p.utensils = lambda state: self.ut
        self.s = NS(timestep=1, players=[NS(held_object=None), NS(held_object=None)], objects={})
        self.p.ask = lambda prompt: (_ for _ in ()).throw(ValueError('bad JSON'))

    def old(self, i, g):
        self.p.typed[i], self.p.cached[i] = g, sentence(g)

    def hold(self, i, item):
        self.s.players[i].held_object = NS(name=item)

    def test_completed_fetch_not_replayed(self):
        self.old(1, goal('get_from_dispenser', 'egg'))
        self.hold(1, 'egg')
        self.assertIn('shared counter', self.p.goals(self.s)[1])
        self.assertEqual(self.p.calls[-1]['retired'][1], 'completed')

    def test_completed_deposit_not_picked_back_up(self):
        self.old(1, goal('place_on_counter', 'egg'))
        self.s.objects['egg'] = NS(name='egg', position=(2, 1))
        goals = self.p.goals(self.s)
        self.assertIsNone(goals[1])
        self.assertIn('pick up', goals[0])
        self.assertEqual(self.p.calls[-1]['retired'][1], 'completed')

    def test_finished_food_advances_to_serve(self):
        self.old(0, goal('take_out', 'boiled_egg', 'pot0'))
        self.hold(0, 'boiled_egg')
        self.assertIn('serve', self.p.goals(self.s)[0])

    def test_valid_incomplete_goal_preserved(self):
        g = goal('get_from_dispenser', 'egg')
        self.old(1, g)
        self.assertEqual(self.p.goals(self.s)[1], sentence(g))
        self.assertEqual(self.p.calls[-1]['sources'][1], 'retained')

    def test_failure_is_not_intentional_idle(self):
        self.p.goals(self.s)
        self.assertEqual(self.p.calls[-1]['sources'][1], 'fallback')
        self.assertIsNotNone(self.p.calls[-1]['err'])
        self.s.timestep += 5
        self.p.ask = lambda prompt: dict(reasoning='nothing', chef=goal('idle'), assistant=goal('idle'))
        self.assertEqual(self.p.goals(self.s), {0: None, 1: None})
        self.assertEqual(self.p.calls[-1]['sources'][1], 'intentional_idle')

    def test_malformed_response_falls_back(self):
        for raw in ({}, [], {'reasoning': 'bad', 'chef': None, 'assistant': goal('fly')},
                    {'reasoning': 'bad', 'chef': goal('get_from_dispenser', 'egg'), 'assistant': goal('fly')},
                    {'chef': goal('idle'), 'assistant': goal('idle')}):
            with self.subTest(raw=raw):
                self.s.timestep += 5
                self.p.ask = lambda prompt, raw=raw: raw
                self.assertIn('egg', self.p.goals(self.s)[1] or '')
                self.assertIsNotNone(self.p.calls[-1]['err'])

    def test_completion_precedes_preconditions(self):
        cases = [(goal('put_into', 'egg', 'pot0'), dict(status='full', contents=['egg'], product=None)),
                 (goal('start', station='pot0'), dict(status='cooking', contents=[], product='boiled_egg')),
                 (goal('wait', station='pot0'), dict(status='ready', contents=[], product='boiled_egg'))]
        for g, ut in cases:
            with self.subTest(g=g):
                self.s.timestep += 5
                self.ut['pot0'] = ut
                self.old(0, g)
                self.p.goals(self.s)
                self.assertEqual(self.p.calls[-1]['retired'][0], 'completed')


if __name__ == '__main__':
    unittest.main()
