# Labeling Copilot

A deep research agent for automated data curation in computer vision.

Labeling Copilot plans, retrieves, annotates, and augments image datasets end-to-end. It composes three model services (retrieval, annotation, synthesis), coordinates them through a top-level agent, and produces COCO-formatted output with per-model detection provenance.

**Authors:**
Debargha Ganguly<sup>\*1,3</sup>, Sumit Kumar<sup>\*1,4</sup>, Ishwar Balappanawar<sup>\*1,4</sup>, Weicong Chen<sup>\*3</sup>, Shashank Kambhatla<sup>1,5</sup>, Srinivasan Iyengar<sup>2</sup>, Shivkumar Kalyanaraman<sup>2</sup>, Ponnurangam Kumaraguru<sup>4</sup>, Vipin Chaudhary<sup>3</sup>

<sup>1</sup>Microsoft Research &nbsp;&nbsp; <sup>2</sup>Microsoft Corporation &nbsp;&nbsp; <sup>3</sup>Case Western Reserve University &nbsp;&nbsp; <sup>4</sup>IIIT Hyderabad &nbsp;&nbsp; <sup>5</sup>University of Pennsylvania

<sup>\*</sup>Equal contribution.

---

## Architecture

| Component | Role | Port |
|---|---|---|
| `retrieval/` | CLIP-indexed image pool with active-learning samplers, backed by MinIO + FAISS | `8000` |
| `annotate/` | Multi-model detection (Detic, OWL-ViT, GroundingDINO) and segmentation (SAM, SEEM) behind a single orchestrator | `8080` |
| `synthesis/` | On-demand image augmentation via vision-LM suggestions + image-edit generation, with OOD filtering | `8090` |
| `viewer/` | Lightweight web UI for browsing curated images, detections, and gap analysis | `8501` |
| `agent_demo.py` | End-to-end orchestration example | — |

## Prerequisites

- NVIDIA GPU with recent drivers
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker compose` v2+
- Python 3.10+ on the host

Sanity-check GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
git clone https://github.com/ICICLE-ai/labelingcopilot.git
cd labelingcopilot

./scripts/docker-up.sh --with-synthesis   # recommended: full stack (retrieval + annotate + synthesis)
./scripts/docker-up.sh                    # retrieval + annotate only
./scripts/docker-up.sh --run-agent        # start services, then run the demo agent
./scripts/docker-down.sh                  # stop everything
```

We recommend running with `--with-synthesis`. The agent uses synthesis to fill scene-coverage gaps it identifies during curation; without it, those gaps can only be addressed by sampling more from the existing pool. Synthesis needs vision-model API credentials — see [Synthesis setup](#synthesis-setup) below for the one-time configuration.

The helper script validates Docker and GPU access, auto-selects a runtime profile for your host, builds the model containers, and waits for each service to report healthy. First startup downloads model weights and datasets into Docker volumes, so expect several minutes of setup on a cold machine.

Verify the services:

```bash
curl http://localhost:8000/health   # retrieval
curl http://localhost:8080/health   # annotate
curl http://localhost:8090/health   # synthesis (if started)
```

## Using Labeling Copilot

### Recommended: drive it through a coding agent

Labeling Copilot is a curation *loop* over three tools — retrieve, annotate, synthesise. Running that loop well means planning the next batch, parallelising I/O, interpreting partial failures, tagging scenes, deciding when to trigger synthesis, and persisting state across sessions. That orchestration layer — the "harness" described in the paper — is a hard engineering problem in its own right: context-window management, tool-call sequencing, retry strategy, memory compaction, and recovery from ambiguous intermediate outputs.

Mature coding agents (Claude Code, Codex, and similar) already solve the harness problem. We therefore expose Labeling Copilot's capabilities as plain HTTP services plus a markdown skill file, and let the agent drive them. For any real curation run this is the recommended path.

With [Claude Code](https://docs.claude.com/en/docs/claude-code):

```
/curate curate 100 images of cats and dogs with diverse lighting
```

The skill at `.claude/commands/curate.md` is loaded automatically. It runs an iterative loop with active-learning sampling, multi-model detection, gap analysis, and targeted synthesis, persisting state to `agent_output/` so you can stop and resume.

### Why not MCP?

The same tools could be exposed as an MCP server. We chose not to: modern LLMs invoke shell and HTTP calls very efficiently through their native tool use, and an MCP layer adds schema metadata into every turn's context window without changing what the agent can actually do. Plain HTTP endpoints plus a markdown skill file keep the context small and the surface universal — any agent framework can `curl` them.

### Minimal reference: `agent_demo.py`

The repo also ships a short, single-file Python program:

```bash
python agent_demo.py
```

It runs a fixed cat/dog pipeline — sample, auto-label, detect, save — and is useful for two things: verifying your stack is wired correctly, and reading as an example of the underlying tool calls. It is **not** an agent. It has no planning, no backtracking, and no memory across runs, because reproducing the harness inside a single Python event loop is exactly the engineering problem we recommend delegating. For anything beyond a smoke test, use `/curate` or point your own coding agent at the HTTP services.

### Extending to your data, classes, or models

Cat vs dog is a motivating example, not a ceiling. See [EXTENDING.md](EXTENDING.md) for a walkthrough: changing detection vocabulary, swapping the image pool, writing a custom ETL for your own data, adapting the curation taxonomy, and adding new model services or samplers.

## Viewer

The viewer renders curated images with bounding-box overlays, per-model scores, scene coverage, and gap analysis. It auto-refreshes so you can watch a curation run unfold.

```bash
python viewer/server.py --port 8501
```

Open <http://localhost:8501>.

## Synthesis setup

Synthesis generates targeted image variants to fill scene-coverage gaps identified during curation. It requires vision-model API credentials (OpenAI or Azure OpenAI), so it is not started by default.

```bash
cp synthesis/.env.example synthesis/.env
# edit synthesis/.env — set either OPENAI_API_KEY or the AZURE_* vars
./scripts/docker-up.sh --with-synthesis
```

See [`synthesis/README.md`](synthesis/README.md) for the service overview, [`synthesis/API.md`](synthesis/API.md) for endpoint reference, and [`synthesis/TROUBLESHOOTING.md`](synthesis/TROUBLESHOOTING.md) for common issues.

## Configuration

The retrieval service loads images on startup from the dataset named in `retrieval/docker-compose.yml`. Ships with `cat,dog` from Oxford-IIIT Pets; edit `LABEL_CLASSES` to change.

GPU builds are controlled by a small set of environment variables consumed by the top-level script and Dockerfiles:

| Variable | Purpose |
|---|---|
| `CUDA_BASE_IMAGE` | Base CUDA image for the annotation stack |
| `PYTORCH_PACKAGES` | Torch packages to install (e.g. `torch==2.5.1 torchvision==0.20.1`) |
| `PYTORCH_INDEX_URL` | Optional pip index (e.g. `https://download.pytorch.org/whl/cu121`) |
| `TORCH_CUDA_ARCH_LIST` | CUDA arches to compile for |
| `INSTALL_TORCH` | Set to `0` to skip torch install (for containers that ship torch) |

On `aarch64` / Blackwell-class hosts, the script auto-selects a CUDA 13 profile. On common x86_64 hosts it selects a CUDA 12.1 profile. Set any variable above to override.

## Repository layout

```
labelingcopilot/
├── annotate/        # detection + segmentation orchestrator
├── retrieval/       # CLIP pool + active-learning samplers
├── synthesis/       # augmentation API
├── viewer/          # web UI
├── scripts/         # docker-up / docker-down
├── agent_demo.py    # end-to-end demo
├── docker-compose.agent.yml
├── LICENSE
└── README.md
```

## Contributing

We welcome pull requests. Labeling Copilot is designed to be extended, and there are several directions where community contributions would be especially valuable:

- **Additional annotation tools** — new detectors or segmenters wired behind the orchestrator. The model-service pattern is documented in [EXTENDING.md §5](EXTENDING.md). Open-vocabulary detectors, specialist medical or remote-sensing models, and lightweight CPU-friendly baselines are all welcome.
- **Additional retrieval tools** — new samplers (diversity, uncertainty, coreset, budgeted), alternative feature backbones (DINOv2, SigLIP, custom embeddings), or swap-in vector stores. See [EXTENDING.md §6](EXTENDING.md).
- **Open-source replacements for paid components** — synthesis currently calls hosted vision and image-edit APIs. Drop-in backends using open-weight models (e.g., Llava/Qwen2-VL for suggestions, SDXL/FLUX for image edits via ComfyUI or diffusers) would let the full pipeline run offline. The provider abstraction in `synthesis/utils/api_client.py` is the right place to plug these in.
- **New ETL loaders** — helpers for common dataset formats (COCO, YOLO, Open Images, LabelBox/CVAT exports) in `retrieval/`.
- **New curation taxonomies** — alternative scene dimensions in `.claude/commands/curate.md` for domains beyond photographs of objects (documents, medical imaging, remote sensing, industrial inspection).
- **Bug fixes and docs** — always appreciated.

Please open an issue before large changes so we can align on scope. Small fixes can go straight to a PR.

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{ganguly2025labeling,
  title={Labeling copilot: A deep research agent for automated data curation in computer vision},
  author={Ganguly, Debargha and Kumar, Sumit and Balappanawar, Ishwar and Chen, Weicong and Kambhatla, Shashank and Iyengar, Srinivasan and Kalyanaraman, Shivkumar and Kumaraguru, Ponnurangam and Chaudhary, Vipin},
  journal={arXiv preprint arXiv:2509.22631},
  year={2025}
}
```

## Contact

Correspondence: `debargha@case.edu`.

Full author contacts: `{debargha, weicong, vipin}@case.edu`, `{sriyengar, shkalya}@microsoft.com`, `sumit.k@research.iiit.ac.in`, `ishwar.balappanawar@students.iiit.ac.in`, `pk.guru@iiit.ac.in`, `skamb@seas.upenn.edu`.
