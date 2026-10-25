# Mini Metro research verification

Inspected with Chrome DevTools on 2026-10-25. This is a source audit, not a reproduced gameplay result. No packages installed, checkpoints loaded, training run, or adapter implemented.

## Bottom line

No specific latest/state-of-the-art Mini Metro paper was verified. Do not attribute a best-performing Mini Metro approach to this research. Search coverage is incomplete: Google, DuckDuckGo and Brave presented bot challenges; Bing returned irrelevant Mini car results. arXiv's all-field `mini metro` query returned one unrelated optical-network paper; Crossref's title query returned unrelated transport/city records. These negative searches do not prove no paper exists.

Useful primary-source artifacts do exist: a Python simulator with a recurrent PPO implementation/design note, and a separate simplified fork with PPO checkpoint files. Start with a simulator/checkpoint feasibility probe, not a reproduction of an unidentified paper.

## 1. Python simulator and architecture recommendation

Source: https://github.com/yanfengliu/python_mini_metro (main displayed ef1be2d during inspection).

- README describes pygame-ce implementation, both human and program input, MIT license, Python 3.13 environment, Gymnasium pixel environment, optional PyTorch/SB3/SB3-Contrib training dependencies.
- README provides random/noop/demonstrator/model recording commands. It reports Classic random: 0 deliveries / 364 decisions; demonstrator: 20 / 958. These are author-reported sanity baselines, not independently reproduced scores or learned-policy results.
- A detailed design note recommends ten-frame multiscale CNN + 256-unit LSTM + RecurrentPPO for its pixel-only lane. It explicitly says: “These are hypotheses to test, not claims that one architecture is universally best. No published benchmark covers this exact game, rendering protocol, or action schema.”
- Note specifies RGB observations, default 10 Hz decision cadence, and MultiDiscrete([8,192,108]) action-kind/pointer coordinates. Semantic structured inputs are explicitly a distinct assisted task, not equivalent to pixel-only play.
- DreamerV3 is a deferred research lane requiring new adaptation, not an established Mini Metro winner. Structured/semantic recurrent policies are proposed diagnostics, not demonstrated superior results.
- README contains significant optional recursive-playtest/Node tooling. Do not assume all of it is necessary for a minimal Python environment probe, or execute remote setup instructions without inspection.

Architecture note: https://github.com/yanfengliu/python_mini_metro/blob/main/docs/rl-model-selection.md

## 2. Simplified fork with actual checkpoint files

Sources:
- https://github.com/Samuel-LY-Wang/RL_Mini_Metro (main displayed 5893dd7)
- https://github.com/Samuel-LY-Wang/RL_Mini_Metro/tree/main/checkpoints
- https://raw.githubusercontent.com/Samuel-LY-Wang/RL_Mini_Metro/main/RL/RL_env.py
- https://raw.githubusercontent.com/Samuel-LY-Wang/RL_Mini_Metro/main/RL/Display_Agent.py

README describes a simplified game and headless RL environment, MIT license. Checkpoints directory visibly contains `chkpt_16384.pth` and `ppo_ckpt.pth`. Existence is verified; quality, compatibility, provenance, and safe load behavior are not.

Source inspection:
- Gymnasium Dict observations: passenger counts, station shapes, route station masks, train location/progress, score, terminal flag. This is structured state, not pixel-only gameplay.
- Seven action kinds: noop, start/add/finish/remove/abort route, atomic whole-route creation. Whole-route construction is assisted macro control, not equivalent to a human pointer gesture protocol.
- Default world step 100 ms; max_steps=1000. Reward combines score changes with boarding/connectivity shaping and action costs. Do not compare shaped return directly with raw delivered-passenger totals from the other repository.
- `rgb_array` rendering raises NotImplementedError; a pygame display helper exists separately.
- Display helper runs the policy to create routes, then switches to a display loop that advances the mediator. It does NOT keep calling the policy during that display loop. Therefore its viewer alone is not proof of continuous reactive AI management under pressure.
- Script uses `torch.load`; do not blindly load untrusted serialized artifacts. Inspect format and use appropriate restricted loading/sandboxing before any execution.

This is the strongest found lead for trying existing weights without new training, NOT a validated strong baseline or paper reproduction.

## 3. Other discovered repositories

- https://github.com/unleex/mini-metro-rl: inspected root contains only README, LICENSE and .gitignore; not a runnable solution as inspected (d57a25f).
- https://github.com/utilForever/InfinityMetro: description says C++ RL simulator but visible tree is Rust/Cargo with a small Python GUI; README offers no demonstrated trained-policy result. Not preferred for a quick Python integration.

Discovery: https://api.github.com/search/repositories?q=mini-metro+reinforcement+learning

## 4. Verified general research, not Mini Metro evidence

DreamerV3: Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap, **Mastering diverse control tasks through world models**, Nature 640, 647–653, published 2 April 2025.

Primary paper: https://www.nature.com/articles/s41586-025-08744-2
Official code linked by paper: https://github.com/danijar/dreamerv3

Publisher text describes learning a recurrent world model and training actor/critic on imagined latent trajectories; reports results across over 150 tasks and Minecraft diamond collection from scratch. Inspected article text contains no `Mini Metro` mention. These results do not establish a Mini Metro checkpoint or state of the art, nor direct reuse of a pretrained general game-playing policy. Algorithm reuse still needs environment adaptation and task training. No claim made that this is the latest general RL paper.

## Recommendation for this project

Do not recreate an RL training pipeline for the hackathon. If exploring Mini Metro, make the first deliverable a bounded feasibility answer:
1. Pin and inspect a Python simulator in isolation; run its tests and basic no-op/render path.
2. Verify a short deterministic route can transport passengers and that timing/seed/reward behavior is measurable.
3. If testing supplied PPO weights, load cautiously and evaluate actual continuous episodes; do not mistake its route-preview viewer for a reactive policy.
4. Expose a small set of legal route operations to existing Jev/Clef and compare with a simple heuristic on identical seeds/time budgets. Label structured-state/macro-action assistance explicitly. This proposed integration requires no new model training, but its quality is unproven.
5. Keep Collab as the working demo until that probe succeeds. No Mini Metro implementation or async Collab work was performed during this research.

## Search provenance

- https://arxiv.org/search/?query=mini+metro&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- https://api.crossref.org/works?query.title=mini%20metro&rows=10&select=title,DOI,published,URL

Limited search cannot support “latest,” “best,” or “no papers exist.”
