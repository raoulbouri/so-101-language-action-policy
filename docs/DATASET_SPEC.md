# Dataset specification

Format: a single **HDF5** file, one group per episode. The layout follows the
standard ACT / LeRobot HDF5 convention, so an existing ACT dataloader needs only
the two extra language keys.

## File layout

```
/metadata                                     (attributes only)
/episode_000000/
    obs/
        scene_image            (T, H, W, 3)  uint8    wide workspace camera
        wrist_image            (T, H, W, 3)  uint8    eye-in-hand camera
        qpos                   (T, 6)        float32  observed joint angles, rad
        qvel                   (T, 6)        float32  observed joint velocities
        tcp_pose               (T, 7)        float32  xyz + wxyz quaternion
    action                     (T, 6)        float32  target joint positions
    action_chunk               (T, k, 6)     float32  k-step lookahead
    action_chunk_mask          (T, k)        bool     False where padded
    phase                      (T,)          int8     index into metadata/phases
    object_poses               (T, n, 7)     float32  every object's xyz + quat
    language_instruction       scalar        utf-8    the raw command string
    language_embedding         (D,)          float32  frozen text encoder output
/episode_000001/
    ...
```

`H, W = 240, 320` by default; `k = 32`; `D = 512` for CLIP ViT-B/32.
`T` varies per episode (≈ 430 steps ≈ 8.6 s at 50 Hz).

## Joint order

Index 0–5 throughout, for both `qpos` and `action`:

```
0 shoulder_pan   1 shoulder_lift   2 elbow_flex
3 wrist_flex     4 wrist_roll      5 gripper
```

The action space is **target joint positions in radians**, fed directly to the
model's position actuators — the same signal a position-controlled real SO-101
takes.

## Frame alignment contract

```
at record index t
    obs[t]     state observed BEFORE anything is commanded at step t
    action[t]  the target joint position commanded at step t
    obs[t+1]   the result of applying action[t]
```

`action[t]` is always the action taken *from* `obs[t]`. Control and recording run
at the same 50 Hz so there is no resampling between the streams and no
opportunity for an off-by-one lag. Asserted in `tests/test_alignment.py`.

## Action chunking

`action_chunk[t] = action[t : t+k]`. Where fewer than `k` real actions remain,
the tail repeats the final action and `action_chunk_mask[t, j]` is `False` for
those entries — repeating keeps padded targets on the manifold, and the mask lets
the loss ignore them entirely.

## Language conditioning

- `language_instruction` — e.g. `"take the red cube and place it in the green circle"`.
- `language_embedding` — a fixed-size L2-normalised vector from a **frozen**
  pre-trained text encoder, constant for a given string across the whole dataset.

`metadata/text_encoder` names the encoder actually used. **Check it**: if
`transformers` was unavailable the pipeline falls back to a deterministic
*hashing* encoder whose vectors carry no semantics (see RISK-004).

For ACT, pass `language_embedding` as a conditioning token into **both** the CVAE
encoder and the transformer decoder, alongside the visual tokens.

## `/metadata` attributes

| Attribute | Meaning |
| --- | --- |
| `fps` | Recording rate, 50.0 |
| `action_chunk_size` | `k` |
| `n_dof` | 6 |
| `image_height`, `image_width` | Camera resolution |
| `text_encoder` | Encoder name, or `hashing-fallback` |
| `language_embedding_dim` | `D` |
| `phases` | The seven phase names, indexed by `phase` |
| `action_space` | Human-readable description |
| `frame_alignment` | The contract above, in the file |
| `n_episodes` | Episodes written |
| `episodes_attempted`, `episodes_successful` | Expert yield |

## Per-episode attributes

`seed`, `instruction`, `success`, `success_reason`, `episode_length`,
`target_object`, `target_object_label`, `target_zone`, `target_zone_label`,
`object_names`, `object_labels`, `zone_labels`, `zone_positions`, and the full
evaluation record as `eval_*` (`eval_center_distance`,
`eval_max_corner_distance`, `eval_cube_height`, `eval_cube_speed`,
`eval_center_in_zone`).

Distractor objects and zones are recorded, not just the target — that is what
makes the dataset a genuine *language-grounding* problem rather than a single
memorised motion.

## Reading an episode

```python
import h5py

with h5py.File("data/so101_lang_act.hdf5") as f:
    print(dict(f["metadata"].attrs))
    ep = f["episode_000000"]
    instruction = ep["language_instruction"][()].decode()
    embedding   = ep["language_embedding"][:]      # (512,)
    images      = ep["obs/scene_image"][:]         # (T, 240, 320, 3)
    qpos        = ep["obs/qpos"][:]                # (T, 6)
    chunks      = ep["action_chunk"][:]            # (T, 32, 6)
    mask        = ep["action_chunk_mask"][:]       # (T, 32)
```
