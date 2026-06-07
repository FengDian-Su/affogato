# External dependencies

These third-party repos are used by the grounding / rendering stages. They are **not**
committed into this repo (large + upstream-managed; see `.gitignore`). Clone them into the
paths below at the pinned commits to reproduce the environment.

| Path           | Repo                                              | Pinned commit | Used by |
|----------------|---------------------------------------------------|---------------|---------|
| `sam2/`        | https://github.com/facebookresearch/sam2.git      | `2b90b9f`     | stage-2 mask grounding (Molmo points → SAM2 masks) |
| `richdreamer/` | https://github.com/modelscope/richdreamer.git     | `e467759`     | multi-view RGB / normal / depth rendering for vis  |

```bash
git clone https://github.com/facebookresearch/sam2.git   && git -C sam2        checkout 2b90b9f
git clone https://github.com/modelscope/richdreamer.git  && git -C richdreamer checkout e467759
```

Our own Molmo + SAM integration code (committed here): `code/infer_molmo.py`,
`code/download_molmo.py`, `data_generation/notebook/molmo_sam_grounding_*.ipynb`,
`prompt/molmo_prompt_format.md`.

Large data/artifacts are also gitignored and regenerable from the pipeline code:
`dataset/` (gObjaverse + affogato GT), `data_generation/{outputs,checkpoints,review,assets,results}/`.
