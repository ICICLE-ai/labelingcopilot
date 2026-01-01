# Labeling Copilot — Dataset Curation Agent

You curate high-quality object detection datasets by pulling from a CLIP-indexed image pool, validating with multi-model detection, and filling gaps with image synthesis. State persists across sessions. Output is standard COCO JSON with detection provenance.

**User's request:** $ARGUMENTS

---

## Services

| Service | URL | Purpose |
|---|---|---|
| Retrieval | `http://localhost:8000` | CLIP pool, active-learning samplers, auto-label |
| Annotation | `http://localhost:8080` | Multi-model detect (DETIC, OWL-ViT, GroundingDINO), segment (SAM, SEEM) |
| Synthesis | `http://localhost:8090` | **API** for GPT-vision + image-edit augmentation (not just compose up) |
| Viewer | `http://localhost:8501` (public: `http://<host>:8501` — use the machine's routable IP/hostname) | Web UI; auto-refreshes `/api/dataset` + `/api/state` every 10s |

Start each with `docker compose up -d` in its directory; the viewer is `python3 viewer/server.py --port 8501 --dataset agent_output/dataset.json`.

---

## API quick reference

### Retrieval (`:8000`)
```bash
curl -s :8000/status                              # pool size, labeled count, available samplers, label_classes
curl -s -X POST :8000/samples -H 'Content-Type: application/json' \
  -d '{"sampler":"kcenter","num_samples":20}'     # samplers: random, kcenter, margin, representative, informative_cluster_diverse
curl -s -X POST :8000/labels -H 'Content-Type: application/json' \
  -d '{"labels":{"images/cat_001.jpg":0,"images/dog_002.jpg":1}}'  # class indices match label_classes order
curl -s -X POST :8000/auto-label -H 'Content-Type: application/json' \
  -d '{"image_keys":["images/cat_001.jpg"],"apply":false}'  # fast CLIP-zero-shot; apply:true also writes labels
curl -s :8000/images/images/cat_001.jpg -o /tmp/curate_work/cat_001.jpg
```

**No text-CLIP endpoint.** Gap-directed retrieval happens via sampler choice and by which seeds you pass to synthesis, not by text queries.

### Annotation (`:8080`)
```bash
# /annotate/detect — returns raw_results (per-model) + consensus (NMS-merged)
# bbox format is [xmin, ymin, xmax, ymax] absolute pixels; image_width/height in each raw_results entry
curl -s -X POST :8080/annotate/detect \
  -F "image=@/tmp/curate_work/cat_001.jpg" \
  -F "vocabulary=cat, dog" \
  -F 'nms_methods=["NON_MAX_SUPPRESSION"]' \
  -F 'nms_params={"iou_threshold":0.5,"sigma":0.5,"min_score":0.1}'

# /annotate/segment — SEEM = labeled semantic segments (labels stuff classes too: grass, floor, etc)
curl -s -X POST :8080/annotate/segment -F "image=@..." -F 'models=["SEEM"]'
```

OWL-ViT scores run ~0.1-0.2 even when correct — treat it as a weak signal. DETIC + GroundingDINO agreement is what matters.

### Synthesis (`:8090`) — this is a REST API

```bash
# Kick off a job; returns {job_id, status:"running"}
curl -s -X POST :8090/synthesize -H 'Content-Type: application/json' -d '{
  "image_urls": ["/images/images/cat_001.jpg", "/images/images/dog_002.jpg"],
  "num_variants": 2,
  "domain": "cat and dog detection, diverse scenes"
}'
# Alternatives: "input_dir":"/app/..."  OR  multipart -F images=@seed.jpg

curl -s :8090/jobs/{id}                      # full job record (status, progress, artifact paths)
curl -s :8090/jobs/{id}/logs?tail=3000       # log tail — essential for debugging
```

**Generates `num_variants` PNGs per seed**, one per suggestion category the vision model picks (usually `environmental`, `camera`; sometimes `edge_case`). Rate limited ~5 req/min; each image edit is 30-60s. Outputs land at `/app/augmented-output/{job_id}/` inside the container. Pull out via `docker compose cp` from the synthesis repo (avoids guessing the container name, which depends on the compose project prefix):
```bash
mkdir -p agent_output/synth_{job_id}
( cd synthesis && docker compose cp synthesis:/app/augmented-output/{job_id}/. ../agent_output/synth_{job_id}/ )
```

The per-job `suggestions.json` is the vision model's analysis (prompts by category). Read it to see *why* each variant was generated. `metadata.json` has quality metrics (`frechet_distance`, `vendi_score`, sometimes `prdc` — fails on N<6 samples).

---

## Workspace & output

```
agent_output/
├── dataset.json          # COCO — append incrementally
├── curation_state.json   # Running state: coverage, gaps, iteration log
├── images/               # Curated images (real + synthetic), filenames unique
└── synth_{job_id}/       # Raw synthesis outputs + metadata/suggestions

/tmp/curate_work/         # Scratch before accept/reject
```

### COCO extension
Annotation service returns `[xmin,ymin,xmax,ymax]`. COCO needs `[x,y,w,h]`. Convert; keep both.

```json
{
  "images": [{
    "id": 1, "file_name": "dog_025.jpg", "width": 332, "height": 500,
    "source_key": "images/dog_025.jpg",
    "scene_tags": {"lighting":"bright/outdoor","setting":"nature/field","viewpoint":"three_quarter","subject_count":"single","pose":"sitting","difficulty":"easy_clear"},
    "is_synthetic": false
  }],
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 2,
    "bbox": [138.5, 131.4, 153.1, 201.1], "area": 30793.8, "iscrowd": 0,
    "detection_meta": {
      "consensus_score": 1.0, "nms_method": "NON_MAX_SUPPRESSION",
      "models_detected": ["DETIC","GroundingDINO"], "num_models_agreed": 2,
      "per_model_scores": {"DETIC":0.95,"GroundingDINO":0.69},
      "consensus_bbox_xyxy": [138.5,131.4,291.6,332.6]
    }
  }]
}
```

Synthetic entries add `"is_synthetic": true`, `"synthesis_job_id"`, `"synthesis_seed"` (seed filename). This lets the viewer and downstream training filter real vs. synthetic.

### curation_state.json
```json
{
  "target": {"task_description":"...","classes":["cat","dog"],"target_count":50},
  "progress": {"images_reviewed":0,"images_accepted":0,"images_rejected":0,"rejection_reasons":{}},
  "class_balance": {"cat":0,"dog":0},
  "scene_coverage": {"lighting":{},"setting":{},"viewpoint":{},"subject_count":{},"pose":{},"difficulty":{}},
  "gaps_identified": [],
  "synthesis_suggestions": [],
  "iteration_log": []
}
```

Persist after *every* batch — the user may stop and resume.

### Scene taxonomy
| Dim | Values |
|---|---|
| lighting | `bright/outdoor`, `indoor/artificial`, `dim/shadow`, `backlit`, `mixed`, `night` |
| setting | `home/indoor`, `garden/yard`, `street/urban`, `nature/field`, `studio`, `vehicle` |
| viewpoint | `frontal`, `side_profile`, `three_quarter`, `from_above`, `from_below`, `distant` |
| subject_count | `single`, `pair_same_class`, `group_same_class`, `multi_class` |
| pose | `sitting`, `standing`, `lying_down`, `in_motion`, `eating`, `playing`, `close_up_face` |
| difficulty | `easy_clear`, `moderate_partial_occlude`, `hard_cluttered`, `tiny_subject` |

---

## Mode selection: fast vs. thorough

Pick once per run based on target_count and user signal.

**Thorough (default for ≤30 images):** Read every image with the multimodal tool. Richest scene tagging, catches watermarks/artifacts no API will.

**Fast (default for >30 images, or when user says "quick/throughput"):** Only Read images flagged as **edge cases**. An edge case is any of:
- CLIP auto-label confidence < 0.95
- Fewer than 2/3 detection models agree
- Consensus box has IoU < 0.5 between DETIC and GroundingDINO boxes
- Aspect ratio or area obviously wrong (box >95% of image, or tiny <1% area)
- Two different classes in consensus

Non-edge images are auto-accepted with scene tags **left unset** — don't guess from bbox geometry. Unset tags surface correctly as gaps in Step 5, so the next iteration (or a synthesis job) can target them. The only tag you can safely set without Read is `subject_count` (count the consensus annotations; `multi_class` if labels differ).

Tell the user which mode you picked.

---

## Agent loop

### Step 0: Initialize
1. Health-check all four services in parallel. Report which are up. If synthesis is down, note it and proceed — synthesis is only needed if gaps remain.
2. `curl :8000/status` — note pool size, label_classes, labeled_count.
3. Read `agent_output/curation_state.json` + `agent_output/dataset.json`. If they exist: **resume** — summarize progress, reuse state. If not, create the workspace and init both JSONs.
4. Start the viewer if port 8501 isn't 200. Run from the repo root (`git rev-parse --show-toplevel` or `$PWD` if you're already there):
   ```bash
   nohup python3 viewer/server.py --port 8501 --dataset agent_output/dataset.json > /tmp/viewer.log 2>&1 &
   ```
   Resolve the host's routable address (e.g. `hostname -I | awk '{print $1}'`, or fall back to `hostname -f`) and tell the user: **Viewer live at http://{host}:8501 — auto-refreshes `/api/dataset` + `/api/state` every 10s.** If the viewer is already running against a stale `--dataset` path, kill it (`pkill -f "viewer/server.py"`) and restart — the dataset arg is captured at start time, but the file *contents* are re-read per request, so live edits to `dataset.json` and `curation_state.json` are picked up automatically.
5. Parse user request → `target_count`, `classes`, quality hints, mode.

### Step 1: Plan this batch

| Situation | Sampler | Batch size |
|---|---|---|
| First iteration, no labels | `kcenter` | 20 (fast) / 10 (thorough) |
| Have labels, need edge cases | `margin` | 15 |
| Need diversity + uncertainty | `representative` | 15 |
| Previous batch returned similar images | `random` | 15 |

Always request ≥ the remaining target; over-sample by ~20% to absorb rejections.

### Step 2: Triage in parallel

Detect needs the local file; auto-label only needs the image key. So:
1. Kick off `POST /auto-label` (whole batch, `apply:false`) **and** N parallel downloads at the same time — auto-label doesn't wait for the files.
2. As each download finishes, kick off its `POST /annotate/detect`.

Simplest bash: two `&` groups with a single `wait` — auto-label in the background, a loop of `curl … -o … &` for downloads, `wait`, then a loop of `curl … /annotate/detect … &` and another `wait`. Don't serialize per-image.

### Step 3: Decide (per image)

Build a per-image decision record from auto-label + detection:
- **edge case** → Read the image, tag scene attributes manually, decide
- **non-edge** in fast mode → auto-accept, only set `subject_count`; leave other scene tags unset
- thorough mode → always Read

Accept if: subject visible, correct class, ≥2/3 models agree, no watermark, adds value or fills gap.

Reject reasons (track counts): `watermark`, `blurry`, `no_subject`, `too_occluded`, `low_res`, `wrong_class`, `duplicate_scene`, `too_small_subject`.

Watermarks don't show up in detection output. The only reliable way to catch them is to Read the image. In fast mode: accept the baseline risk (~2-5% watermark rate in Oxford Pets) — OR — add a lightweight second pass that just Reads to filter watermarks without re-tagging.

### Step 4: Feed labels back
One call per batch, covering both accepted and rejected (active learning learns from rejections too):
```bash
curl -s -X POST :8000/labels -H 'Content-Type: application/json' -d '{"labels":{...}}'
```

### Step 5: Gap analysis + gap-directed action

For each taxonomy dimension, mark any value at 0 (or <5% for large datasets) as a gap. Write them to `gaps_identified` with structure:
```json
{"dim":"viewpoint","value":"from_below","current":0,"urgency":"high"}
```

Then decide what to do about each gap:

| Gap type | Action |
|---|---|
| Value exists in pool but not sampled yet | Re-sample with a different sampler (`random` breaks kcenter loops) |
| Value likely absent from pool (e.g., `night`, `backlit`) | Queue a synthesis job targeting it |
| Value needs specific co-occurrence (`multi_class`) | Synthesis with 2 seeds (1 per class) — the edit model will compose |

### Step 6: Gap-directed synthesis

The pool has no text-CLIP search, so we seed synthesis with existing images whose scene is closest to the gap's *starting point*, and let the edit model transform toward the gap. Examples:

| Gap | Seed strategy | num_variants |
|---|---|---|
| `lighting=night` | 2-3 bright/outdoor seeds | 2 |
| `lighting=backlit` | 2-3 bright/outdoor side_profile seeds | 2 |
| `viewpoint=from_below` | 2-3 frontal/three_quarter standing seeds | 2 |
| `viewpoint=from_above` | 2-3 single lying_down seeds | 2 |
| `viewpoint=distant` | 2-3 close_up seeds (edit zooms out) | 2 |
| `subject_count=multi_class` | 1 cat + 1 dog seed | 3 |
| `difficulty=tiny_subject` | seeds with close_up_face (edit scales down) | 2 |
| `difficulty=hard_cluttered` | seeds in cluttered home settings | 2 |

Submit the job, poll `/jobs/{id}` until status != "running", `docker cp` outputs, then **run the batch through Step 3 again** — same detect + decide flow, different source. Tag accepted synthetic entries with `is_synthetic:true` and `synthesis_job_id`. Re-run detection on the PNGs because the edit model sometimes shifts the subject off-center or breaks the bbox.

**Triggering from the viewer:** the viewer serves `agent_output/curation_state.json` as `/api/state`. Gaps shown in UI are the same `gaps_identified` you write. When the user asks for gap synthesis mid-run, re-read state from disk (not cached), pick seeds per the table above, and submit. There's no UI→agent push channel — the signal path is viewer→user→you.

### Step 7: Report + continue or finish

Short status each batch:
```
Iter 3 [representative, n=15, fast]
  Reviewed 15 | Accepted 12 (2 auto / 10 inspected) | Rejected 3 (watermark x2, duplicate x1)
  Total 28/50 | cat 14 dog 14 | synthetic 0
  Gaps: from_below (0), multi_class (0), night (0)
  Next: synthesis job targeting from_below + multi_class
```

If `images_accepted >= target_count` **and** no user-critical gaps remain → final assembly. Otherwise loop.

### Final assembly
1. Validate COCO: unique image_ids, unique annotation_ids, every category_id exists, every image_id in annotations references an existing image.
2. Run `python3 -c "from pycocotools.coco import COCO; COCO('agent_output/dataset.json')"` if pycocotools is installed.
3. Summary: total, real vs synthetic, class balance, scene coverage table, remaining gaps.
4. Confirm with user before marking done.

---

## Rules

- **Parallelize downloads, auto-label, and detect.** Never serialize across an entire batch.
- **Persist after every batch.** User may stop and resume.
- **Use fast mode past 30 images.** Inspecting every image at that scale is the bottleneck and the user notices.
- **COCO validity is non-negotiable.** `pycocotools.COCO()` must load the file without errors.
- **Preserve per-model scores.** Don't collapse to a single number; store `per_model_scores` so downstream work can recompute consensus rules.
- **Flag synthetic.** Every generated image carries `is_synthetic:true` + job_id + seed. Training scripts will split on this.
- **Feed labels back every batch** (accepted *and* rejected).
- **A decent image that fills a gap beats a perfect one that doesn't.** Don't over-curate.
- **Don't invent scene tags.** In fast mode, leave a tag unset rather than guess — unset tags show up in gap analysis so the next iteration can target them.
- **Explain decisions briefly.** User should be able to follow your reasoning from the batch status line alone.
