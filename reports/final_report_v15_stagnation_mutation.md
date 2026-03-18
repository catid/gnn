## APSGNN v15: Stagnation-Gated Mutation

### What Changed

v15 tested one narrower mutation gate on top of the existing `querygrad` utility selector:

- keep utility-based parent selection unchanged
- keep mutation only in the final `24 -> 32` transition
- keep the high score margin gate `mutation_score_margin = 0.75`
- add a new requirement that the previous stage's validation tail be flat before mutation is allowed

The stagnation gate used:

- `mutation_require_stagnation: true`
- `mutation_stagnation_window: 2`
- `mutation_stagnation_delta: 0.02`

Interpretation: mutation is allowed only if the last two stage-tail validation points are within `0.02` query accuracy.

### Why This Follow-Up

v11-v14 showed a consistent pattern:

- unconditional mutation could improve some late-stage metrics, but transfer was weaker
- stricter confidence-only gates still did not make mutation a robust default
- the next plausible trigger was "mutate only when the current stage looks stalled"

This was meant to test whether mutation works better as a recovery mechanism than as a default branching mechanism.

### Exact Core Regime

The benchmark stayed matched to the existing longplus selective-growth setup:

- final compute leaves: `32`
- output node: `0`
- train writers per episode: `2`
- eval writers per episode: `2, 6, 10`
- start node pool size: `2`
- query TTL: `2..3`
- max rollout steps: `12`
- active-node schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`
- total steps: `8000`
- stage steps: `[250, 250, 300, 300, 400, 600, 5900]`
- bootstrap at each stage start: `75` steps
- visible GPU count used: `2`

Bootstrap packets were excluded from task-only coverage and utility accounting, consistent with v8-v14.

### Runs Completed

Completed core runs:

- [20260318-124259-v15-utility-querygrad-stagnate-longplus-s1234](/home/catid/gnn/runs/20260318-124259-v15-utility-querygrad-stagnate-longplus-s1234)
- [20260318-124936-v15-utility-querygrad-stagnate-longplus-s2234](/home/catid/gnn/runs/20260318-124936-v15-utility-querygrad-stagnate-longplus-s2234)
- [20260318-125613-v15-utility-querygrad-stagnate-longplus-s3234](/home/catid/gnn/runs/20260318-125613-v15-utility-querygrad-stagnate-longplus-s3234)

Interrupted after enough evidence to stop:

- [20260318-130250-v15-utility-querygrad-stagnate-longplus-s4234](/home/catid/gnn/runs/20260318-130250-v15-utility-querygrad-stagnate-longplus-s4234)

The interrupted fourth seed was not counted in the summary statistics.

### Core Comparison

| Arm | Seeds | Best val | Last val | K2 | K6 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 `querygrad` | 4 | `0.6927 ± 0.0578` | `0.5677 ± 0.0400` | `0.6133 ± 0.0531` | `0.5195 ± 0.0545` | `0.4863 ± 0.0347` |
| v13 high-confidence mutation | 4 | `0.6823 ± 0.0462` | `0.5833 ± 0.1227` | `-` | `0.5234 ± 0.0506` | `0.5137 ± 0.0117` |
| v14 component-agreement mutation | 4 | `0.6823 ± 0.0574` | `0.6094 ± 0.0521` | `-` | `0.5176 ± 0.0483` | `0.5059 ± 0.0173` |
| v15 stagnation-gated mutation | 3 | `0.6736 ± 0.0547` | `0.5764 ± 0.0428` | `0.6250 ± 0.0567` | `0.5286 ± 0.0603` | `0.5052 ± 0.0133` |

### Per-Seed v15 Results

| Seed | Best val | Last val | K2 | K6 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1234` | `0.6250` | `0.6250` | `0.6797` | `0.5547` | `0.4922` |
| `2234` | `0.6458` | `0.5208` | `0.5469` | `0.4453` | `0.5000` |
| `3234` | `0.7500` | `0.5833` | `0.6484` | `0.5859` | `0.5234` |

### Interpretation

v15 is a real data point, but not a win. The stagnation gate kept the mutation branch competitive and prevented the clear transfer problems seen in the earlier unconditional variants, but it still did not beat plain `querygrad` convincingly on the metrics that matter most for promoting a new default.

The strongest evidence is:

- best validation stayed below `querygrad` mean
- last-checkpoint stability was only roughly tied with `querygrad`
- `K6` was slightly better than `querygrad`, but not by enough to outweigh the lower best-val mean
- `K10` stayed in the same band as the prior mutation-gated baselines, not clearly above them

That means stagnation is a somewhat better mutation trigger than broad mutation, but still not a strong enough policy to justify further rollout or transfer budget.

### Decision

No H1 transfer pair was run for v15.

Reason:

- the completed core seeds were competitive
- but they were not clearly better than the mutation-free `querygrad` baseline
- so additional transfer compute would likely have repeated the same conclusion at higher cost

### Conclusion

The current default should remain utility-only `querygrad` selective growth.

v15 strengthens the existing conclusion rather than changing it:

- mutation can be made less harmful with stronger gating
- but even a plausible stagnation trigger still does not produce a robust upgrade over mutation-free selective growth

### Outputs

- summary JSON: [summary_metrics_v15.json](/home/catid/gnn/reports/summary_metrics_v15.json)
- smoke run: [20260318-124136-v15-utility-querygrad-stagnate-smoke-s1234](/home/catid/gnn/runs/20260318-124136-v15-utility-querygrad-stagnate-smoke-s1234)
- core runs: [runs](/home/catid/gnn/runs)

### Next Best Move

If mutation is revisited again, the next defensible idea is not another static gate. The sharper next experiment is an adaptive mutation trigger driven by a stronger confidence signal or explicit late-stage stagnation plus utility margin, with mutation remaining off by default.
