# Documentation index

| File | What it is | Read it when |
| --- | --- | --- |
| [PROGRESS.md](PROGRESS.md) | Component-by-component status against the directive, plus the build log and what is explicitly out of scope | You want to know what is done and what is not |
| [ISSUES.md](ISSUES.md) | Every bug found, its root cause, the measurement that proved it, and the fix. Plus open risks | Something looks wrong, or you are about to change a physical constant |
| [DECISIONS.md](DECISIONS.md) | Design choices with their alternatives and evidence | You are wondering "why is it done this way" |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module map, data flow, the seven phases, coordinate conventions | You are new to the code |
| [DATASET_SPEC.md](DATASET_SPEC.md) | The exact on-disk HDF5 schema and the frame-alignment contract | You are writing a dataloader |

## How to read the issue log

Every entry follows the same shape: **symptom → investigation → root cause →
fix → effect**. The investigation section is the valuable part; it records the
measurement that distinguished the real cause from the plausible one. Three of
the five issues were caused by a physically plausible number being assumed
rather than measured, and none of them crashed anything — they showed up as a
degraded success rate or, worse, as silently degraded data quality.

Two lessons generalise beyond this repo:

- **A metric cannot see dataset quality.** ISSUE-004 (camera framing, shadow
  acne) had zero effect on the success rate and would have degraded every
  training image. It was found by watching the video.
- **Check machine load before trusting a performance number.** ISSUE-005 nearly
  produced a confident, wrong root cause because orphaned worker processes had
  pushed the load average to 28.
