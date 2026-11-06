# GAWorld Repository Hygiene Audit

**Status:** read-only audit (TASK-3218, RESULT-3218) restated here as the actionable
hygiene record. **No cleanup has been executed** — every action below is a proposal.
Labels: **Verified** (checked on disk / in git), **Computed** (derived from measurements),
**Proposed** (recommended, not done).

## 1. Inventory summary

**Verified.**

- Repo at 2.0 GB on disk; `.git` 248 MB (243.85 MiB pack, 591 commits, no LFS).
- **4,379 tracked files; 0 untracked non-ignored files; ~67,924 ignored files.**
- Dominant on-disk consumers: `.venv-fos-integration/` **856 MB** (ignored), `fos/` 431 MB
  (incl. `fos/frontend/node_modules` **374 MB**, ignored), `.git` 248 MB, `output/` 117 MB
  (ignored), `site/` 54 MB, `docs/` 53 MB (incl. `docs/proposals/results/` 36 MB of
  **tracked** experiment outputs).

**Computed.** Pack-bloat drivers (largest blobs, incl. dead history):
52 MB (`FOS_0621 2.pptx`, still at HEAD), ~46 MB and ~22.5 MB (earlier pptx revisions,
history-only), 20.2 MB (mp4), 14.1 MB. Removing files from HEAD will **not** shrink
`.git`; only a history rewrite would (see §7 step 8).

## 2. Secrets: CLEAN

**Verified.** Pattern scan over all tracked files (`sk-…`, `AKIA…`, `BEGIN … PRIVATE KEY`,
`ghp_…`, long api-key literals): **zero hits**. `.env` does not exist (only
`.env.example` placeholders). `dashboard_config.json` carries `api_key_env` **names**,
not values. `gaworld/llm/credentials.py` refuses in-repo secrets and requires
`GAWORLD_LLM_SECRETS_PATH` outside the repo — good posture. No flagged files.

## 3. Hygiene findings

**Verified.**

- **Tracked agent scratch (6):** `task_plan.md`, `progress.md`, `findings.md`,
  `read.doc` (binary Word doc containing only a URL note), `readme_wsy.md` (personal run
  report), `plan-with-files.skill`. Six more `TASK-*.progress` files are untracked
  (correctly ignored) and stale.
- **Tracked build artifacts (3):** `fos/src/fos.egg-info/{PKG-INFO,dependency_links.txt,top_level.txt}`.
- **Tracked outputs (1):** `outputs/019ed11e-…/…/gaworld-project-introduction-zh-en.pptx`
  — an output-artifact dir mirroring the ignored `output/` naming, one letter off.
- **Tracked-despite-ignored (1):** `dashboard_config.json` — runtime-mutated by
  `gaworld.city use`; the .gitignore rule is inert for tracked files.
- **Duplicate PNG (2 files, byte-identical):** `video/public/gaworld-graphical-abstract.png`
  ≡ `site/assets/gaworld-graphical-abstract.png` (2.58 MB each). Triple screenshot copies
  across `site/home/docs/_static|_images` and `site/documentation/_static`.
- **Ignored-but-not-covered gap:** `fos/frontend/dist/` (build output) is untracked only
  by luck; `outputs/` (plural) has no ignore rule.
- **Profiler dumps tracked:** `docs/perf/baseline.prof`, `docs/perf/post-perf-fix.prof`.

## 4. Large tracked files (all >1 MB; 24 files >1 MB, 3 >10 MB)

**Computed.**

| File | Size | Assessment |
|---|---|---|
| `FOS_0621 2.pptx` | 52,062,455 | junk — copy-numbered one-off deck; biggest liability |
| `site/home/media/gaworld-intro-zh-music.mp4` | 20,226,601 | website asset; candidate for release-asset hosting |
| `GAWorld_教学手册_预览.pdf` | 11,337,064 | deliverable; keep `.md` source tracked, binary out |
| `ppt_assets/GAWorld_项目介绍.html` | 9,044,884 | presentation export; keep at most one version |
| `GAWorld_教学手册.docx` | 5,966,236 | deliverable binary |
| `site/home/assets/classical-engraving.png` | 3,536,875 | site asset |
| `paper/arxiv…/main.pdf` + `-source.tar.gz` | 3.0 MB + 2.8 MB | paper artifacts; keep or release-asset |
| `site/home/assets/cultural-montage.png` | 2,818,408 | site asset |
| `textbook/textbook-complete.zip` | 2,775,213 | redundant (pdf+md tracked) |
| `video/public/gaworld-graphical-abstract.png` | 2,584,489 | **duplicate** — keep one copy |
| `site/assets/gaworld-graphical-abstract.png` | 2,584,489 | duplicate — keep one copy |
| `GAWorld_项目介绍_2026-09.pptx` / `.pdf` | 2.5 MB / 1.7 MB | current deck; keep |
| `docs/perf/baseline.prof`, `post-perf-fix.prof` | 1.7 MB ×2 | profiler dumps — analysis artifacts, not docs |
| `docs/proposals/results/**` | 36 MB, 946 files | experiment run outputs inside docs/ — biggest mislabel |

## 5. MCF docs consolidation map

**Verified.** The MCF phase docs form a clean linear supersession chain:

```
MCF_GAWORLD_INTEGRATION_FEASIBILITY.md (root, living, updated in place)
  → PHASE_0_5_VALIDATION → PHASE_0_6_ZERO_COST_VALIDATION
  → PHASE_0_7_EVALUATION_VALIDATION → PHASE_0_8_OFFLINE_AUDIT
  → PHASE_0_9_PREFLIGHT  (authoritative end of chain; Stage 1 verdict: GO)
  → PHASE_1_0_ENGINEERING_AUDIT (new; corrections + canary spec + revised Stage 1)
```

**Proposed.** Keep **two** authoritative docs: the updated Feasibility doc + Phase 0.9
(+ Phase 1.0 as its correction record). Archive `PHASE_{0_5,0_6,0_7,0_8}*.md` to
`docs/archive/mcf/` or repo-external storage — their surviving content is already
restated in 0.9. **Do NOT delete them**: they are the evidence trail for the scorer's
change record.

## 6. Classification table

**Computed.**

| Category | Count | Action |
|---|---|---|
| Essential source (`gaworld/**`, `fos/src/**`, `scripts/`, entrypoints) | ~2,900 | keep |
| Essential tests (`tests/**`, `fos/tests/**`, `site/*.test.js`) | ~700 | keep |
| Permanent docs (tutorials, references, design docs) | ~120 | keep |
| Internal research reports + run artifacts (`docs/proposals/results/**` 946, `docs/perf`, merge/weekly/refactor reports, `docs/superpowers/**`) | ~1,050 | archive outside repo |
| Agent scratch (`task_plan.md`, `progress.md`, `findings.md`, `read.doc`, `readme_wsy.md`, `plan-with-files.skill`) | 6 | untrack |
| Generated outputs (`fos/src/fos.egg-info/*` ×3, `outputs/**pptx` ×1) | 4 | untrack + ignore |
| Logs/caches/venvs (already ignored) | ~67,900 on disk | no action |
| Duplicates (graphical abstract ×2, screenshot ×3 paths, `.v1.md`, textbook zip, `backup/generative_city_sim_2601*.py` ×4) | ~8 | consolidate |
| Sensitive | **0** | — (§2) |
| Large media keepers (mp4, handbook pdf/docx, paper pdf/tar, ppt_assets) | ~15 | owner decision: keep vs release-asset |

## 7. Proposed cleanup plan (8 steps — NOT executed)

1. **KEEP:** `gaworld/`, `tests/`, `fos/` (minus dist/egg-info), `site/` (minus duplicate
   screenshots), permanent `docs/`, `data/`, `paper/`, `legacy/` (documented deprecation),
   root packaging/config, `textbook/textbook-complete.{md,pdf}`.
2. **UNTRACK** (`git rm --cached`) then archive locally: `FOS_0621 2.pptx`; `outputs/`;
   `fos/src/fos.egg-info/`; `docs/perf/*.prof`; `tmp/imagegen_shims/typing_extensions.py`;
   `task_plan.md`, `progress.md`, `findings.md`, `read.doc`, `readme_wsy.md`,
   `plan-with-files.skill`; `backup/**` (25 files).
3. **IGNORE:** add rules per §8.
4. **ARCHIVE (move out of repo, e.g. `~/GAWorld-archive/`):** `docs/proposals/results/`
   (36 MB) + `docs/proposals/experiments/`; MCF phase docs 0.5–0.8 (per §5, keep 0.9 +
   1.0 + Feasibility in-repo); internal reports (`MERGE_LATEST_DEV_REPORT_2026-09-04.md`,
   `DEV_MAIN_MERGE_REVIEW_2026-10-07.md`, `WEEKLY_REPORT_2026-09-06.md`, `PHASE4_AUDIT.md`,
   `REFACTOR_BASELINE.md`, `REFACTOR_PLAN.md`, `RUN_SIMULATION_EXTRACTION_PLAN.md`,
   `TWIN_DEPLOYMENT_2026-09-16.md`, `TWIN_MOBILE.md`, `PUBLIC_DEPLOYMENT_HANDOFF_2026-09-17.md`,
   `ideas.md`, `todo_board.html`, `docs/superpowers/`).
5. **CONSOLIDATE:** delete `video/public/gaworld-graphical-abstract.png` (keep
   `site/assets/` copy); dedupe the three screenshot paths; pick one root presentation
   set; fold MCF 0.5–0.8 history into the archive (per §5).
6. **DELETE outright (safe):** six stale root `TASK-*.progress` files;
   `fos.db-shm`/`fos.db-wal` (after FOS processes stop); `data/hangzhou_profiles_with_names.v1.md`;
   `textbook/textbook-complete.zip`.
7. **Untrack `dashboard_config.json`** (runtime-mutated, already ignored); ship a
   `dashboard_config.example.json` instead.
8. **OPTIONAL — history shrink (approval required):** `git filter-repo` to drop the
   ~46 MB / ~22.5 MB dead blobs and the 52 MB pptx from history; `.git` 248 MB → well
   under 100 MB. Force-push coordination required — do **not** attempt casually.

**Computed effect:** steps 2–6 remove ~120 MB from the work tree and ~60 MB from
HEAD-tracked content; with step 8 the clone drops from ~2.0 GB / 248 MB git to roughly
~600 MB on disk / <100 MB git (the 856 MB venv is already local-only).

## 8. Recommended .gitignore additions

**Proposed.**

```gitignore
/outputs/
fos/frontend/dist/
*.prof
```

(`*.egg-info/` already covers root but cannot untrack `fos/src/fos.egg-info/` — that
needs step 2's explicit `git rm --cached`.)

## 9. Cleanup risks

**Proposed assessment.**

- **Untracking binaries (step 2)** breaks anyone relying on them from a fresh clone
  (e.g. the pptx decks) — archive first, announce in the change log.
- **`dashboard_config.json` untrack (step 7)** changes first-run behaviour; ship the
  example file and a one-line note in `docs/` before untracking.
- **Archiving `docs/proposals/results/` (step 4)** removes raw evidence some experiment
  reports link to — keep the archive path stable and note it in the reports.
- **History rewrite (step 8)** invalidates all clones and open PRs; requires every
  contributor to re-clone. Highest-risk, lowest-necessity step.
- **Screenshot dedup (step 5)** touches paths referenced by `site/home` HTML — grep for
  references before deleting copies.

## 10. Final recommendations

**Proposed.**

1. Execute steps 2–3 + 6 first (pure wins: junk, build artifacts, agent scratch; zero
   behavioural risk), in one commit.
2. Then steps 4–5 + 7 as a second, reviewed commit (moves + dedup + config example).
3. Defer step 8 (history rewrite) until clone size is an actual operational problem;
   approval-required when it happens.
4. Adopt the §8 ignore rules immediately so the hazards (`outputs/`, `fos/frontend/dist/`,
   `*.prof`) cannot regrow.
5. Re-run this audit's measurement commands after each step to confirm the deltas
   (`git ls-files | wc -l`, `git count-objects -vH`, `du` top-level).

---

*No cleanup executed; no commits; no destructive operations. Sources: RESULT-3218
(read-only audit) + TASK-3220 doc requirement. All figures Verified/Computed as labelled.*
