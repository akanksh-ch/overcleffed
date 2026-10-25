# Mini Metro: deeper literature and method follow-up

2026-10-25, research performed inline with Chrome DevTools. Extends `mini-metro-research.md`; does not replace independently executed simulator feasibility evidence in `mini-metro-spike.md`.

## Revised conclusion

The initial search was too narrow to select a method. A deeper search found **a substantially better-supported related method with public pretrained weights**: graph-neural learned heuristics inside an evolutionary transit-network optimizer. It claims state-of-the-art results on the Mumford transit-network benchmark. This is concrete, reproducible research to consider adapting, **not demonstrated Mini Metro SOTA**.

It is entirely reasonable to implement/reuse a published method rather than invent a policy. But matching its original benchmark, adapting it to a new game, and testing Jev/Clef are three different experiments. None should silently stand in for the others.

No definitive best commercial Mini Metro agent was established. Semantic Scholar exact-name and game/RL queries yielded mostly unrelated transport literature. Google Scholar blocked access. Broader GitHub discovery found additional actual-game projects, but none inspected supplied a trustworthy common-benchmark superiority comparison.

## Strongest verified related research: neural evolutionary transit planning

**Andrew Holliday, Ahmed El-Geneidy, Gregory Dudek — Learning Heuristics for Transit Network Design and Improvement with Deep Reinforcement Learning.**

Primary paper: https://arxiv.org/abs/2404.05894
Full text inspected: https://arxiv.org/html/2404.05894
Journal reference shown by arXiv: *Transportmetrica B: Transport Dynamics*, 13(1), 2025.
DOI: https://doi.org/10.1080/21680566.2025.2561863

Version provenance: original submission 8 April 2024; inspected default full text identifies v6, revised 26 October 2025. Do not mislabel current v6's PPO/results as necessarily present in original v1, or interpret search-index dates as publication dates. These displayed version dates precede this 2026 review.

### Method and evidence

- Trains a graph neural policy with PPO to construct routes by composing shortest-path segments, alternating extension and stopping decisions.
- Uses the policy to initialize a network and as a route-replacement heuristic inside an evolutionary improvement algorithm: the **Neural Evolutionary Algorithm (NEA)**.
- Evaluates Mandl/Mumford benchmarks and a Laval case study. Abstract claims new Mumford SOTA and up to19% simulated cost savings against Laval's existing network. Introduction reports up to4.8% improvement over compared methods on challenging benchmarks. These are paper-reported results, not our reproduction and not Mini Metro scores.
- Important counterexample: its random shortest-path-composition heuristic (RC-EA) beats its neural heuristic in the passenger-travel-time-only objective. Even in this paper there is no universally best policy across objectives.
- Inputs/objective involve a road/link graph, travel times, origin-destination demand, route-count/length constraints, and passenger/operator cost trade-offs. This differs from Mini Metro's stochastic arrivals, station unlocks, fleet capacity, live queues, and losing condition.

### Public code and weights verified in DevTools

Code: https://github.com/aholliday/transit_learning (inspected root commit39656ba).
License: GPL-3.0. Reuse must respect licensing; do not silently copy its code into an MIT project.

README documents:
- `learning/eval_route_generator.py` for learned construction;
- `learning/bee_colony.py` for EA/NEA (historical filename retained);
- `neural_bco_mumford` configuration for NEA;
- author-estimated training3–6hours on a modern commercial GPU (not measured by us);
- known mixed-node-count batching issue;
- published benchmark dataset and model-weight links.

Verified directory: https://cim.mcgill.ca/~mrl/projs/transit_learning/ppo_2025/
Lists ten files `inductive_gae_seed_0.pt` through `_9.pt`, each approximately1.2MB, last-modified22February2025. Availability of directory entries verified; actual downloads, deserialization safety, runtime compatibility, and performance NOT tested.

This is stronger evidence for reusable learned route planning than a generic Dreamer recommendation. It still requires task adaptation; small weight files are not proof of cheap inference/end-to-end search or zero-shot transfer.

Thesis expanding this line: https://arxiv.org/abs/2502.17758 — *Applications of deep reinforcement learning to urban transit network design*, Andrew Holliday, submitted25February2025; abstract supports hybrid neural/metaheuristic planning, not Mini Metro.

## Newer related graph-attention PPO research

**Bibek Poudel and Weizi Li — Learning to Design City-scale Transit Routes**, arXiv2512.19767v1,21December2025.

Primary full text: https://arxiv.org/html/2512.19767v1

- End-to-end graph-attention PPO constructs routes node-by-node, using incremental topological rewards and terminal simulation rewards.
- Bloomington dataset:143nodes,243edges,16routes. Uses UXsim with transit extensions; four training configurations,2000episodes each.
- Paper reports25.6% higher service rate and30.9% shorter wait time than existing routes in one high-transit-adoption setting. Other settings/metrics have trade-offs; this is not superiority over all neural evolutionary approaches.
- Inspected text says code/data are public, but no project repository hyperlink was exposed by the inspected HTML. Therefore availability of a matching runnable release/checkpoint remains unverified.
- More recent does not mean a proven stronger or more immediately reusable Mini Metro method.

## Actual Mini Metro projects newly found

1. https://github.com/KuroPhoenix/Mini-Metro-AI-Agent (cda7234)
   - NYCU Spring2025 course project, not a verified peer-reviewed SOTA paper.
   - Describes screenshot perception, RF-DETR/Roboflow, DAgger/behavior cloning, DQN and heuristic pieces.
   - README explicitly warns expert input capture cannot yet map user movements correctly and an external perception model was due to expire5June2025. Detection precision claims do not establish game-playing strength. PR80 is an aim, not a demonstrated leaderboard result.
   - Repo says MIT but dataset section restricts assets to research/education; inspect exact licenses before reuse.

2. https://github.com/Ttopiac/AI-Mini-Metro (8b74482)
   - Simplified Python simulator fork and optimization runner.
   - Source https://raw.githubusercontent.com/Ttopiac/AI-Mini-Metro/main/src/agent_play.py repeatedly generates/evaluates candidate layouts for3000iterations, then applies the best. Not evidence of a trained continuous controller or SOTA.

3. https://github.com/matthesvoss/Mini-Metro-AI (de6f0dd)
   - MIT Java GUI/layout-solver project; README labels work in progress, no benchmark result inspected.

4. https://github.com/ababhhaa/mini_metro_rl (eaf07b9)
   - README describes emergency controller + weekly MCTS, but inspected root mainly exposes vision/template code. No verified complete planner or performance evidence. Do not interpret aspirations about world records as achieved records.

5. https://github.com/KajetanBilinski/MiniMetroAI (325bfaa)
   - C#/BepInEx game bridge and MCP project, with latest visible commit about pause control. No README benchmark result inspected. Potential actual-game access lead, not a validated strong policy.

Discovery/index queries:
- https://api.github.com/search/repositories?q=%22mini+metro%22+AI&per_page=50
- https://www.semanticscholar.org/search?q=%22mini%20metro%22&sort=relevance
- https://www.semanticscholar.org/search?q=Mini%20Metro%20game%20reinforcement&sort=relevance
- https://www.semanticscholar.org/search?q=reinforcement%20learning%20transit%20network%20design%20routes&sort=relevance

## Recommendation and experiment boundaries

- For a **published-method reproduction**, use the transit_learning code + provided weights on its original Mandl/Mumford task first. No new training is inherently necessary. Verify dependency/runtime, artifact safety, and licensing before execution.
- For **Mini Metro**, a bounded adapter feasibility probe must establish demand representation, route legality, update budget, fleet behavior and live-state revalidation. Do not assume the transit benchmark policy can just be plugged in.
- For **showcasing Jev/Clef**, keep it as an explicitly separate chooser, or compare it to the research policy. A GNN/EA that supplies all strategy while Jev merely repeats its top move does not demonstrate Jev's strategic value.
- Current user approval is for the coding pane's bounded existing-Clef versus heuristic continuous-control test: seed42,120simulated seconds/150wall seconds per run,16ms fixed physics,5second decision opportunities, at most24model calls, one outstanding request,3second response deadline, no hidden fallback or service changes. That task has been sent to the interactive coding pane; no results asserted here.

No training/checkpoint execution or Mini Metro product-code change was performed by this research pane.
