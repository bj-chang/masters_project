# progress log

week by week record of what i've done on the msc project so far (mesh dependence in pde-constrained optimisation, supervised by david ham). pieced together from git history, file timestamps, meeting notes and chat history. dates that look approximate are the meeting week the work is anchored to.

read alongside:
- `notes/next_meeting_briefing.tex` (briefing i wrote 4/7/26 for the four-item plan meeting)
- `preliminary_report/preliminary_report.tex` (3 page report submitted 6/7/26)
- `dissertation/main.tex` (the actual dissertation)

---

## week of 23/3 to 29/3/26 (project start)

- 24/3/26 first meeting with david
- project scoped: reproduce schwedes, ham, funke, piggott (2017) chapter 2 iteration count experiment in the firedrake + pyadjoint + petsc tao stack. their table 2.2 measures l-bfgs iterations on the poisson optimal control problem across non-uniform meshes
- initial reading agreed:
- schwedes chapter 1 (riesz representation theorem, frechet derivatives, adjoint equations, function-space steepest descent / newton-cg / bfgs)
- schwedes chapter 2 (the experiment being reproduced)
- no code or dissertation content in git yet

---

## april / early may 26 (reading and scoping, pre-git)

- extended reading:
- schwedes chapter 1 in full: riesz representation theorem (thm 1.7 in the book), discrete riesz map $\mathcal{R}_{L^2} = M_h^{-1}$ (eq 1.125), function-space l-bfgs (sect 1.5.4.1), newton-cg in hilbert space (sect 1.5.3)
- schwedes chapter 2: the mesh-dependence result, sect 2.2 for the analytical bound, sect 2.3 for the experiments on deterministic (sect 2.3.1) and random-refined (sect 2.3.2) meshes
- hinze, pinnau, ulbrich, ulbrich (2009) as the standard pde-constrained optimisation reference, used for the adjoint derivations
- kirby (2010) siam review 52(2):269-293, cited twice by schwedes as the source for the krylov version of the riesz-map idea. this is what motivated the inner-cg preconditioning experiment that became the novel result later
- some initial code experiments locally but nothing committed

---

## week of 25/5 to 31/5/26 (start of git, reframing)

- 27/5/26 meeting with david
- matt knepley on the petsc discord had said "firedrake wants to switch from rol to tao". david echoed this in the meeting. reframing: not just reproducing schwedes, more about giving firedrake the empirical evidence and interface design for the switch
- three linked goals for the dissertation:
- empirical validation: reproduce panel (a) of table 2.2 through firedrake + pyadjoint + petsc tao. find the settings that make tao lmvm match moola's flat row. extend across the wider tao family
- design question: from david's meeting note "provide mechanism with which correct preconditioner can work, design condition of how automatic it would be". what should pyadjoint's taosolver expose so users get correct mesh-independent behaviour by default?
- preconditioning at two levels: schwedes' riesz map argument applies twice. at the outer l-bfgs via the initial hessian seed (as schwedes does), and at the inner krylov of newton-cg via a preconditioner (schwedes points at this via kirby but doesn't run the experiment). this is the novel contribution
- 28/5/26 initial git commit (`108d762`, "initial commit: dissertation and code"). dissertation and a first code layout under version control. code structure organised into `meshdep/` (library), `experiments/` (top-level scripts), `investigations/` (diagnostic scripts), `test/` (correctness)

files that existed at this point (or were being drafted):
- `dissertation/main.tex`: the dissertation manuscript. at this point a skeleton with sections 1 and 2 (introduction and mathematical preliminaries) starting to fill in
- `dissertation/biblio.bib`: bibliography
- `code/meshdep/__init__.py`: package marker with a short docstring
- `code/meshdep/meshes.py`: mesh constructors. uniform, from file, tensor-product exp-stretched (fallback), netgen. also `graded_unit_square_from_file()` which loads a `.msh` and reports realised $h_{\max}/h_{\min}$
- `code/meshdep/optimisers.py`: three wrappers. `solve_with_tao` (pyadjoint's taosolver for lmvm), `solve_with_scipy` and `solve_with_scipy_external_check` (scipy l-bfgs-b with an external convergence check), `solve_with_hilbert_lbfgs` (custom l-bfgs with every dot product in the hilbert inner product; the reference flat-row implementation)
- `code/meshdep/problems/__init__.py`, `code/meshdep/problems/forward_poisson.py`, `code/meshdep/problems/poisson_control.py`: poisson optimal control setup, desired state $d = \sin(\pi x) \sin(\pi y)$, tikhonov regularisation
- `code/meshdep/problems/burgers.py`, `code/meshdep/problems/burgers_control.py`: 1d viscous burgers control, used for the non-quadratic problem check
- `code/experiments/poc_uniform.py`: proof of concept on uniform meshes
- `code/experiments/table_2_2.py`: the main panel (a) reproduction script. runs (mesh, optimiser, riesz_map) combos and prints iteration counts
- `code/mesh_generation/generate_meshes.py`: script that generates the six netgen graded `.msh` files (target $r \in \{4, 8, 16, 32, 64, 128\}$)

---

## week of 1/6 to 7/6/26 (lmvm history-size bug)

- discovery phase. tao's lmvm at default settings failed to reproduce moola's flat row on graded meshes
- 2/6/26 commit (`1449c10`), "minimal reproducer for lmvm mesh dependence issue on graded meshes". started `code/investigations/minimal_reproducer.py` to isolate the bug
- 3/6/26 commit (`5433265`), "fixed the historical size capped at 5 issue"
- finding: petsc tao/lmvm defaults its history size to 5. on strongly graded meshes the rank-5 quasi-newton update doesn't capture enough of the discrete hessian spectrum. lmvm stagnates above the discrete optimum
- fix: register the option globally before constructing taosolver

```
PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)
```

- passing the option via pyadjoint's `parameters` dict does NOT work because of an options-prefix mismatch between pyadjoint and petsc's internal lmvm mat. now documented in the dissertation section 10.4 as the pyadjoint options-database propagation paragraph
- several sanity checks written this week:
- `code/investigations/test/test_burgers_lmvm.py` (7/6): does h1 alone survive on burgers with default lmvm history? testing whether the fix generalises off poisson
- `code/investigations/test/test_burgers_nls.py` (7/6): does nls misbehave on a nonlinear problem? per josh hope-collins's point that poisson is quadratic in m, so exact-newton lands the minimum in one step regardless of mesh
- `code/investigations/test/test_resolution.py` (7/6): per josh, plot the optimised solution at various grading levels to check whether the high-stretch meshes have effectively turned the problem into a different one
- `code/investigations/test/test_norm_trajectory.py` (7/6): per lawrence mitchell's question, monitor both the $L^2$ and $\ell^2$ norms of the gradient at every outer iteration, to check whether tao's internal convergence test is using the wrong norm
- `code/investigations/test/test_lmvm_h0_rescaling.py` (7/6): demonstrates that pyadjoint's taosolver disables l-bfgs initial-hessian rescaling in certain configs. added a retrospective note: the behaviour is real but was later superseded by the history-size fix as the main cause of the lmvm issue
- diagnostic story so far, two competing hypotheses ran:
- hypothesis 1: noise floor (lmvm stagnating below adjoint-tape round-off). ruled out by running tao/nls at $\varepsilon = 10^{-30}$ and seeing $\|R_{L^2}(J')\|_{L^2} \approx 10^{-18}$ on every mesh, well below the $10^{-5}$ where lmvm stagnated
- hypothesis 2: limited-memory approximation. confirmed by the history-size fix working. rank-5 quasi-newton update too coarse for the graded hessian spectrum

---

## week of 8/6 to 14/6/26 (tao solver zoo and h1 experiments)

- 8/6/26 commit (`5386ec1`), "redone code layout". package structure cleanup
- 9/6/26 meeting with david
- 9/6/26 commits (`9435684`, `fdd64cd`), "general changes, up to meeting on 9/6/26" and "tidied zoo code"
- built out the tao solver zoo. every petsc tao solver that accepts a (possibly bound-constrained) unconstrained problem of this class, run on the same poisson problem across the netgen grading sweep. this became table 10.z of the dissertation
- files created/updated this week:
- `code/investigations/test/test_tao_solver_zoo.py`: runs every tao solver on poisson across the netgen grading sweep. produces the table
- `code/investigations/test/test_tao_solver_zoo_burgers.py`: same survey on 1d burgers. later invalidated when supervisor flagged the synthetic exp-stretched interval collapsing cells on the dirichlet endpoint
- `code/investigations/test/test_h1_riesz.py` (10/6): per josh's suggestion, try `riesz_map="H1"` on the control as an alternative to the lmvm history workaround. question: is changing the inner product on the control space alone enough?
- `code/investigations/test/test_nls.py` (10/6): nls on netgen graded poisson control
- `code/investigations/test/test_poisson_lmvm_vs_nls.py` (10/6): side-by-side check that both lmvm (with hist=100 fix) and nls are mesh-independent on poisson control, netgen meshes
- `code/investigations/test/test_poisson_lmvm_vs_nls_stretch.py` (10/6): same on the synthetic exp-stretched meshes. range of realised ratios via stretches $s = 3, 4, \ldots, 12$
- `code/investigations/test/test_burgers_lmvm_vs_nls.py` (9/6): same on 1d burgers
- `code/investigations/test/test_burgers_lmvm_vs_nls_stretch.py` (10/6): same on exp-stretched 1d meshes
- zoo results (dissertation table 10.z):

| tao solver | r=4 | r=16 | r=64 | r=128 | conclusion |
|---|---|---|---|---|---|
| `nls` (newton line search) | 2 | 2 | 2 | 2 | mesh-independent (quadratic problem) |
| `bnls` (bounded newton ls) | 1 | 1 | 1 | 1 | same |
| `bntr` (bounded newton tr) | 8 | 9 | 10 | 10 | mesh-independent |
| `bntl` (bounded newton tr+ls) | 8 | 9 | 10 | 10 | same |
| `lmvm` (hist=100 fix) | 43 | 41 | 41 | 42 | mesh-independent |
| `bqnls` (bounded qn ls, default hist=5) | 33 | 60 | 88 | 118 | mesh-dependent |
| `cg` (nonlinear cg) | - | - | - | - | mesh-dependent, budget-limited |

- `bqnls` isolated as the interesting case. same algorithmic class as `lmvm` (bounded quasi-newton line search) but mesh-dependent at defaults. at the time i wrote this up as the same limited-memory truncation issue (history too small) and used it as supporting evidence that the lmvm history story was the right one. item 4 of the four-item plan much later would refine this considerably

---

## week of 15/6 to 21/6/26 (dissertation writing pass)

- substantial writing on `dissertation/main.tex`. chapters covered:
- chapter on riesz representation and the adjoint method (a restructured version of what david had suggested as section 2 mathematical preliminaries)
- discrete forms and the mass matrix
- mesh-dependence chapter with the tao zoo results and the history-size fix
- implementation notes
- 21/6/26 commit (`c4b0460`), "latex changes up to 21/6/26"
- applied david's writing style rules (from earlier david email, saved in memory as `feedback_dissertation_writing_style.md`):
- first-person-plural-present for derivations ("we derive", "we substitute")
- impersonal, no anthropomorphism for algorithms ("the algorithm applies", not "we apply")
- past tense for experimental results ("we ran", "the counts grew from x to y")
- my own style rules on top: plain student voice, no em dashes, no jargon vocabulary

---

## week of 22/6 to 28/6/26 (inner-cg discovery and riesz pc implementation)

- 23/6/26 meeting with david
- the inner-cg idea. attention turned to the fact that nls's outer count is trivially 2 iterations on poisson (quadratic), so the "nls is mesh-independent" reading of table 10.z was hiding a mesh dependence one level deeper. nls solves $H p = -g$ inside each newton step with an inner krylov (cg). that inner count grows with grading
- `code/investigations/test/test_poisson_monitor.py` (23/6): monitors inner cg iteration count per outer newton step. two concerns:
- david's nls suspicion: "the line search might do dot products in the wrong inner product". trace it and see
- track the inner-cg count so we can quantify the hidden mesh dependence
- riesz map preconditioner implementation started
- `code/meshdep/preconditioners.py` written. two layers of pc:
- `RieszMapL2` and `RieszMapH1`: subclasses of firedrake's `AuxiliaryOperatorPC`. framework-native route. subclass just overrides `form(pc, v, u)` to return the mass form ($L^2$) or mass + stiffness ($H^1$). the framework handles everything else
- `RieszMapPCContext`: plain petsc python pc context, about 50 lines. takes the function space v explicitly at construction. assembles $M_h$ (or $M_h + K_h$ for h1), factorises via lu, applies on every `apply()` call
- why the second class was needed: `AuxiliaryOperatorPC` auto-discovers the function space from the operator matrix. works for a normal firedrake solve, fails silently for tao's inner ksp because pyadjoint's `ReducedFunctionalHessianMat` is a python mat without a function space attached. pc stays at "pc type: none" with no warning
- `code/investigations/test/test_riesz_pc.py` (27/6): first version of the riesz pc sweep. confirmed the options-database route was silently ignored. moved to programmatic attachment via `solver.tao.getKSP().getPC().setPythonContext(...)`
- `code/investigations/random_refine.py` written this week or slightly earlier. implements schwedes sect 2.3.2.1 random rivara refinement. each cell independently marked with probability $p = 0.35$, marked cells get longest-edge bisected, lepp chain propagates through neighbours to keep the mesh conforming. `random_refined_mesh(level, seed=42, ...)` returns a firedrake mesh at any refinement level
- `code/investigations/test/visualise_random_refined_meshes.py` and `code/investigations/test/visualise_netgen_meshes.py`: render the mesh families as pngs for the dissertation and for sanity
- `code/investigations/test/test_poisson_lmvm_vs_nls_random.py` (10/6 originally): lmvm vs nls on the random-refined meshes

---

## week of 29/6 to 5/7/26 (four-item plan wrapped up + preliminary report)

- the four-item action plan came out of the 23/6 meeting. david set out four steps:
- write an $L^2$ riesz-map preconditioner as an `AuxiliaryOperatorPC` subclass
- plumb the preconditioner into tao's `nls` inner krylov via petsc options
- confirm inner-ksp iteration count is mesh-independent under grading
- investigate the `bqnls` line-search trace under grading
- items 1 to 3 completed this week
- `code/investigations/test/test_riesz_pc_sweep.py` (4/7): the headline item 3 script. runs nls on each netgen graded mesh in $r \in \{4, 16, 64, 128\}$ twice
- baseline (no pc on the inner ksp)
- with riesz pc (`RieszMapPCContext(V, riesz_map="L2")` attached programmatically to the inner ksp after taosolver construction). uses `ksp.setConvergenceHistory(reset=False)` so petsc keeps the residual history across both outer newton steps; total inner-cg count is `len(ksp.getConvergenceHistory())` at the end
- item 3 headline results (netgen graded):
- baseline no-pc inner cg: 56, 147, 432, 807 at $r = 4, 16, 64, 128$. roughly doubles per refinement
- with $L^2$ riesz pc: flat at 15 across the whole sweep
- outer newton stayed at 2 iterations in every row
- 54x reduction at $r = 128$
- written up as section 10.8 of the dissertation
- `notes/next_meeting_briefing.tex` (4/7): 11 page briefing document for david laying out the four-item plan, why each step was worth doing, what the outcomes on items 1 to 3 were, and for item 4 exactly what to run and how to read the output. compiled to `next_meeting_briefing.pdf`
- preliminary report drafted this week and submitted 6/7. `preliminary_report/preliminary_report.tex` plus `preliminary_report.bib`. 3 pages. sections:
- introduction and context (rol-to-tao switch framing, firedrake demo citations)
- background and literature (schwedes, kirby, hpuu)
- objectives (three-item framing)
- methods (poisson control setup, mesh sweep, verification)
- preliminary results (panel a reproduction, inner-cg story)
- work plan for the remaining time
- multiple wording iterations before submission (softening the rol-to-tao framing to "several backends including tao", adding firedrake demo urls as footnotes, cutting from 4 pages down to 3, fixing an overstated polynomial-bound claim, correcting a stray `\textt{}` typo)

---

## week of 6/7 to 12/7/26 (current week: bqnls investigation, mesh construction, poster)

### 6/7/26

- preliminary report submitted to supervisor + university inbox (leo.iwasaki@yahoo.co.uk)
- `code/investigations/test/run_random_sweep_for_poster.py` written and run. full poster-data sweep on schwedes random-refined meshes at levels 0, 2, 4, 6, 8, 10. produces the four rows the poster needs:
- scipy l-bfgs-b in $\ell^2$, convergence externally checked in $L^2$ (growing panel-a row)
- hand-written hilbert l-bfgs in $L^2$ (flat panel-a row)
- tao nls baseline no pc (growing inner-cg row)
- tao nls + $L^2$ riesz pc (flat inner-cg row)
- results (whole-domain random-refined meshes):

| level | cells | ratio | scipy $\ell^2$ | hilbert $L^2$ | nls inner base | +pc |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 1.00 | 13 | 6 | 24 | 10 |
| 2 | 340 | 2.00 | 31 | 8 | 39 | 14 |
| 4 | 1027 | 2.83 | 35 | 6 | 45 | 14 |
| 6 | 3174 | 4.00 | 47 | 6 | 50 | 13 |
| 8 | 9670 | 5.66 | 47 | 5 | 61 | 13 |
| 10 | 29474 | 5.66 | 55 | 4 | 73 | 12 |

- observation: realised $h_{\max}/h_{\min}$ caps at ~5.7 across levels 8 and 10 even though the cell count triples. the lepp chain propagation is the cause. once a cell is bisected, its neighbours have to be bisected too (recursively) to keep the mesh conforming, so at $p = 0.35$ across the whole domain, chains eventually refine most of the mesh. $h_{\max}$ shrinks at nearly the same rate as $h_{\min}$
- poster construction started:
- `poster/poster.tex`: a1 portrait using `beamerposter` + `pgfplots`. imperial branding (blue accent, imperial logo png trimmed of padding, matching block heights)
- `poster/gen_mesh_figure.py`: generates mesh figures at chosen levels using firedrake `triplot` + matplotlib
- `poster/gen_comparison_plot.py`: matplotlib log-log plot of inner-cg counts vs realised $h_{\max}/h_{\min}$, comparing netgen graded and random-refined mesh sequences on the same axes to show they trace the same trend curve where they overlap
- imperial logo (`poster/figures/imperial_logo.png`, 600x270 png) imported from downloads. had 109 px of whitespace padding on top/bottom and 55/52 px left/right. `\includegraphics` `trim=55 105 52 105, clip` option strips them

### 7/7/26 (today)

#### item 4: bqnls line-search investigation

- `code/investigations/test/test_bqnls_line_search.py` written. runs `bqnls` on the netgen grading sweep with both `tao_monitor` and `tao_ls_monitor` on. structured `RESULT:` lines emitted so counts can be grepped
- first finding (later retracted): at `-tao_bqnls_mat_lmvm_hist_size 100` via `PETSc.Options().setValue(...)`, iteration counts identical to default (33, 60, 88, 118 at both). read that as the history-size hypothesis being falsified. the ls trace was parsed and looked healthy (no floor hitting, no runaway halving, 1 or 2 ls trials per outer step in most cases)
- retraction via `tao_view`. ran `tao_view` and saw the lmvm matrix reporting `Max. storage: 5` in both runs. the option prefix for bqnls's internal lmvm mat is different from the one i used. `tao_bqnls_mat_lmvm_hist_size` needs to go through pyadjoint's `parameters` dict, not through `PETSc.Options()` with a `-tao_bqnls_...` prefix. both my runs were effectively at hist=5
- corrected sweep. `code/investigations/test/test_bqnls_h0_fix.py` written. passes options through the `parameters` dict correctly. corrected results at hist=100: 19, 36, 59, 76 across $r = 4, 16, 64, 128$. so history does matter (30 to 40% reduction) but doesn't fully close the gap. bqnls still mesh-dependent at hist=100

#### discovery: hypothesis c, missing $M_h$ seed on the lmvm $J_0$

- `tao_view` diff between lmvm-at-100 and bqnls-at-100:
- lmvm: lmvm matrix has $M_h$ installed as $J_0$ (via `pyadjoint.optimization.tao_solver.InitialHessianPreconditioner`, a python-context pc). rescale type: none
- bqnls: lmvm matrix has default diagonal $J_0$. rescale type: diagonal
- same lmvm specialisation (`lmvmbfgs`), same history (100), but the initial-hessian seed is completely different
- read pyadjoint source. `tao_solver.py` line 728:

```python
if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
    ...
    tao.setLMVMH0(B_0_matrix)
```

- bqnls (which is really `TAOBQNLS`, part of the `TAOBNK` family) is not in the set, so the seed install is silently skipped
- petsc-c-level restriction. attempted `tao.setLMVMH0(...)` on bqnls directly. errors:

```
TaoLMVMGetH0KSP() at src/tao/bound/impls/blmvm/blmvm.c:383
This routine applies to TAO_LMVM and TAO_BLMVM.
```

- so the pyadjoint guard mirrors a petsc-side restriction
- `code/investigations/test/test_bqnls_seed_install.py` written. three routes tried to install $M_h$ as $J_0$ on bqnls:
- `solver.tao.getLMVMMat()` then `Mat.setLMVMJ0(python-ctx M_h)`. bypasses the tao-level restriction by operating on the lmvm matrix directly
- same but pass an assembled $M_h$ matrix (not a python context)
- pre-build an entire lmvm matrix with $J_0$ set and swap it in via `setLMVMMat`
- direct causal proof: routes 1 and 2 both work perfectly. route 3 destabilises. results with $M_h$ as $J_0$, hist=100: 19, 18, 23, 18 across $r = 4, 16, 64, 128$. effectively flat. actually faster than lmvm-at-100 (43, 41, 41, 42) because bqnls uses a different lmvm specialisation that converges more aggressively once given the correct seed
- conclusion for item 4: two stacked causes for bqnls's mesh-dependence at petsc defaults:
- default history=5 too small. fix: raise to 100, needs the correct option prefix through `parameters`
- no $M_h$ seed on the lmvm $J_0$. fix: `tao.getLMVMMat().setLMVMJ0(M_h)`
- the line search itself is not misbehaving. david's meeting guess of "non-robustness in the line search" pointed in the right direction (something upstream is wrong) but not the exact mechanism

#### mesh construction investigation

- asked whether the whole-domain random-refined meshes really can't reach netgen-scale $h_{\max}/h_{\min}$ ratios
- `code/investigations/test/test_random_localised.py` written. parameter sweep varying $(p, n_{\mathrm{ini}})$ and refinement level. confirmed: even at $p=0.05, n_{\mathrm{ini}}=4$, ratio tops out around 8 to 11 over 12 levels. the lepp chain propagation is a fundamental limitation
- mathematical reason: uniformly-distributed refinement gives $h_{\max} \approx h_{\min}$ by construction. to reach high ratios, refinement has to concentrate somewhere
- schwedes book re-checked. schwedes table 2.2 (which reaches ratios up to 128) is actually in sect 2.3.1 (deterministically refined meshes using the boundary-doubling scheme of sect 2.2.5), NOT sect 2.3.2 (random-refined). schwedes' table 2.4/2.5 (random) index by refinement level rather than by $h_{\max}/h_{\min}$ and don't explicitly claim ratio 128. so the ceiling isn't in tension with what schwedes actually does. contradicted the earlier assumption i had that random rivara alone reaches those ratios

#### localised random-refined meshes (option 1)

- `code/investigations/test/test_localised_riesz_pc_sweep.py` written. random rivara refinement with marking restricted to a corner sub-square $[0, 0.4]^2$ (matching netgen's fine sub-square). lepp chain still runs but stays contained in the sub-square
- ratios grow by $\sqrt{2}$ per level and reach netgen-scale quickly: level 4 = 4.0, level 8 = 16, level 12 = 64, level 14 = 128
- item 3 results (baseline vs riesz pc) on these meshes:

| level | ratio | cells | baseline | +pc |
|---:|---:|---:|---:|---:|
| 4 | 4.00 | 269 | 63 | 14 |
| 8 | 16.00 | 1461 | 186 | 14 |
| 12 | 64.00 | 12576 | 552 | 14 |
| 14 | 128.00 | 37705 | 973 | 14 |

- story identical to netgen. pc row completely flat

#### interior random-refined meshes (david's preferred variant)

- concern raised that corner refinement includes two dirichlet edges, which might be slightly funny. moved the marking region to the interior $[0.3, 0.7]^2$
- `code/investigations/test/test_interior_riesz_pc_sweep.py` and `code/investigations/test/run_interior_sweep_for_poster.py` written. same sweep on interior meshes
- item 3 results (interior random-refined):

| level | ratio | cells | baseline | +pc |
|---:|---:|---:|---:|---:|
| 4 | 4.00 | 314 | 65 | 14 |
| 8 | 16.00 | 1970 | 197 | 13 |
| 12 | 64.00 | 16020 | 610 | 13 |
| 14 | 128.00 | 46998 | 1111 | 14 |

- baseline slightly stronger than corner version. pc still flat
- full poster-data sweep on interior meshes (scipy + hilbert + item 3) started but interrupted. three of four levels done (level 14 scipy l-bfgs-b was the slow step)
- poster figures added:
- `poster/figures/interior_level4.pdf` (ratio 4, 314 cells)
- `poster/figures/interior_level8.pdf` (ratio 16, 1970 cells)
- `poster/figures/interior_level12.pdf` (ratio 64, 16020 cells)
- `poster/figures/interior_level14.pdf` (ratio 128, 46998 cells)
- also `poster/figures/localised_level{4,8,12,14}.pdf` for the corner variant
- `poster/gen_interior_mesh_figure.py` and `poster/gen_localised_mesh_figure.py` are the generator scripts

#### meeting with david and josh hope-collins (afternoon)

- poster feedback (for the fair, 13/7 deadline):
- too wordy. write about the maths of the code rather than the code itself
- panel (a) point should emphasise "one uses the correct inner product, the other doesn't", not the specific optimiser names
- cut the verification sentence entirely
- fix "by hand" language (it's automatic via pyadjoint)
- mesh-dependence block misses that UNEVEN refinement is the issue, not refinement per se
- do the $H^1$ version of the graph too. probably free
- cut lhs text and add level 0 and 2 meshes to show the progression more clearly
- interior random-refined meshes are a "completely defensible way" of doing it and david is ok with them on the poster
- main next-week technical work: the pyadjoint riesz pc refactor. i should:
- fork the pyadjoint repo, clone, install editably in the firedrake venv
- study the covariance pc pattern in firedrake (josh will show it monday)
- study the existing `RieszMap` factory in `pyadjoint/optimization/tao_solver.py` (line ~487 for `mult`, line ~507 for the `RieszMap` constructor)
- write a new `RieszMapPC` class in the same file with the following interface:
- user writes only two options: `tao_nls_pc_type='python'` and `tao_nls_pc_python_type='pyadjoint...RieszMapPC'`. nothing else needs to be passed
- the pc extracts $V$ and the riesz map choice from `Jhat.controls[0]` (single source of truth, the control already knows its function space and inner product)
- uses petsc's application context (`appctx`) to smuggle `Jhat` from taosolver into the pc
- watch the dual/primal direction: `_ad_init_zero(dual=True)` may need to be primal for the pc to apply the riesz map in the right direction. josh's warning
- aim for eventual merge back into pyadjoint upstream
- dissertation writing: david said spend at least half of remaining time on writing. complete round-trip story: investigated problem, identified cause, delivered solution

---

## week of 8/7 to 14/7/26 (poster submitted, pyadjoint riesz pc built + tested, h1 deep-dive)

### poster (submitted 13/7 for the imperial summer research fair)

- a1 portrait, beamerposter + pgfplots, imperial branding. switched to the interior random-refined mesh figures (david ok'd them for the fair), added level 0 and 2 meshes to show the progression
- applied josh + david's 7/7 feedback: cut the verification sentence, fixed "by hand" -> automatic, rewrote the mesh-dependence block to stress UNEVEN refinement, panel (a) framed as "one uses the correct inner product, the other doesn't"
- later tweaks before submission: reduced jargon for a general audience, justified all the text boxes, enlarged the title, cut the subtitle
- poster presentation scheduled for 21/7 at 13:00

### pyadjoint riesz pc (work stream 2, the main deliverable)

- josh pointed at two templates: firedrake's `CovariancePC` (`firedrake/preconditioners/covariance.py`) and pyadjoint's `RieszMapMatCtx` (`pyadjoint/optimization/tao_solver.py`, class at line 465)
- wrote `RieszMapPC` in `~/pyadjoint/pyadjoint/optimization/riesz_pc.py`. inherits `petsctools.PCBase`. structure copied from `CovariancePC` (`needs_python_pmat=True`, `prefix="riesz"`, initialize/apply/update); the `apply` body and the `dJ` buffer copied from `RieszMapMatCtx.mult`
- the whole point: the user enables it with two options only

```
tao_nls_pc_type = python
tao_nls_pc_python_type = pyadjoint.optimization.riesz_pc.RieszMapPC
```

  and the pc reads the riesz map off each control (`c.riesz_map`) via `self.pmat` (the reduced-hessian mat context). single source of truth, nothing passed by hand
- `code/investigations/test/test_riesz_pc_pyadjoint.py` written. thorough test:
  - l2 sweep on netgen graded meshes r=4,16,64,128: new pc via options matches the reference `RieszMapPCContext` exactly (flat 15) vs baseline (56, 147, 432, 807)
  - h1 spot-check r=16: new pc 223 vs reference 225 (near-match; both use lu, tiny difference)
- archived the original standalone class at `code/investigations/riesz_pc_original.py`

### code changes agreed in the meeting (david + josh)

- rename local `dJ` -> `gradJ` in `apply`. in this bit of maths, derivative = the operator and gradient = the vector; the riesz map takes a derivative and returns a gradient
- narrow the type check from `ReducedFunctionalMatBase` -> `ReducedFunctionalHessianMat`. only the hessian action is primal->dual, so the riesz map (dual->primal) is correct for it. the adjoint (dual->dual) and tlm (primal->primal) mats would give a silent wrong answer, so reject them with a hard error
- wrote the `view` method, modelled on `covariance.py`. prints the pc name + the riesz map per control. the inner-riesz-ksp part isn't shown because the current design shells out to `_ad_convert_riesz` each call rather than holding a persistent solver (david noted this; follow-up)
- verified `view` fires via `tao_view` (shows under the PC Object block; also confirms the pmat is `ReducedFunctionalHessianMat`)

### import-safety issue (open, slack question to david + josh)

- david said to move `RieszMapPC` into `tao_solver.py`. snag: `tao_solver.py` is imported eagerly by `import pyadjoint`, and `petsctools.PCBase` only exists if petsc4py is installed. a plain top-level class would make `import pyadjoint` require petsc4py, regressing josh's lazy-petsc-import work
- options: (1) keep it a standalone module (naturally import-safe, only loaded when the pc is used), (2) lazy factory + module `__getattr__` inside tao_solver, (3) plain top-level class + accept the petsc4py dependency, (4) plain pc class not inheriting `petsctools.PCBase`
- currently kept as the standalone module (option 1). drafted a slack message asking which they prefer before committing

### pr plan (agreed)

- code -> pyadjoint (branch off main). test -> firedrake (branch off release; the test needs firedrake, so it goes in `tests/firedrake/adjoint/test_optimisation.py` ~line 270)
- fork firedrake, checkout release, branch off it, editable install, add the test
- open two prs (pyadjoint + firedrake), ping david + josh on slack. they add the ci fix that points firedrake's ci at my pyadjoint fork branch

### interior mesh refactor

- factored the copy-pasted interior-mesh construction (8 files) into `interior_random_refined_mesh(...)` in `code/investigations/random_refine.py`. behaviour-preserving (mesh sizes reproduce: L=4->314, L=8->1970, L=12->16020 cells)

### h1 deep-dive (why was h1 ~200 iters when l2 is ~15?)

david flagged the high h1 count as looking wrong, probably an artifact, and said to turn the monitors back on and check the tolerances. investigated. found one real artifact and one genuine effect.

- focused h1 sweep (baseline / l2 pc / h1 pc) on interior meshes L=4,6,8,10,12,14 -- `code/investigations/h1_anomaly_sweep.py`
- **artifact: erratic early stops.** the h1 line zigzagged (131, 20, 323, 17, 522, 16). cause (found with `h1_diagnose.py` + `tao_monitor`): `tao_gatol=1e-7` sat right in the per-newton-step residual drop (~1e-7), so at some meshes the optimiser quit after 1 newton step instead of 2. fix: `gatol=1e-10` -> consistent 2 steps -> clean growing line (131, 205, 323, 417, 522, 526). `h1_anomaly_sweep_fixed.py`, graph `figures/h1_fixed.png`
- **checked david's inner-solver hunch -- it wasn't happening.** i first thought firedrake's `RieszMap` defaulted to an iterative inner solve. checked it directly: the default is `preonly+lu` (exact). so there was no inner-solver inflation; every run was already exact. (all my sweeps use either `RieszMapPCContext` = explicit lu, or a `RieszMap(form, solver_parameters=lu)`.)
- **weight sweep.** `h1_weight_sweep.py`. weighted h1 norm M + w*K, swept w = 1, 0.1, 0.01, 0.001, 0.0001. as w->0 the count drops toward l2, but every positive weight still grows; only w=0 (=l2) is flat. graph `figures/h1_weight.png`
- **proved the h1 mesh-dependence is genuine, not an artifact.** same-optimum check (`l2reg_verify.py`): on my l2-reg problem the l2-pc and h1-pc reach the identical minimiser (J* same to 15 digits, m* to ~1e-6) but take 13 vs 323 inner cg at L=8. same answer, different cost = pure conditioning. `tao_monitor` showed clean convergence (residual crashes to 1e-11 in 2 steps, no stagnation)
- **the resolution: the riesz map has to match the regularisation term.** my problem is l2-regularised (alpha int m^2), so the control lives in l2 -> l2 is the correct norm and h1 is the mismatched one -> h1 mesh-dependent. expected, not broken. matches the literature (mesh-independence needs the riesz map of the correct hilbert space; for l2 that's the mass matrix inverse)
- **h1-regularised definitive test.** flipped the regularisation to the h1 seminorm (alpha int |grad m|^2). mirror image: h1 pc flat at 8-10, l2 pc grows 370 -> ~10000+. same-optimum verified again. `h1reg_sweep.py`, `h1reg_verify.py`, graph `figures/h1reg.png`. this is literally david's "find a problem where l2 fails but h1 wins"
- **open point with david.** the plain h1 riesz map (M+K) is mesh-dependent on the l2-reg problem, verified 3 ways (theory, same-optimum experiment, literature). but david said "the problem is in h1". reconciliation: that's true for the state (the pde lives in H1), but the riesz map acts on the control (in L2 here). still need to ask him whether he means the plain h1 riesz map or a smarter matching / sqrt(alpha)-weighted operator preconditioner (schoberl-zulehner type), which could be mesh-robust for l2-reg and i haven't tested

new files this week: `code/investigations/h1_anomaly_sweep.py`, `h1_anomaly_sweep_fixed.py`, `h1_diagnose.py`, `h1_inner_diagnose.py`, `h1_weight_sweep.py`, `h1_weight_sanity.py`, `h1reg_sweep.py`, `l2reg_verify.py`, `h1reg_verify.py`, the matching `plot_*.py` scripts, `riesz_pc_original.py`; figures `figures/h1_anomaly.png`, `h1_fixed.png`, `h1_weight.png`, `h1reg.png` (+ the json data). the class itself lives in the pyadjoint fork at `~/pyadjoint/pyadjoint/optimization/riesz_pc.py`.

---

## full file catalogue

### `dissertation/`
- `main.tex`: the dissertation manuscript. ~1900 lines, bullet-point-heavy in the middle sections (which need converting to prose per david's feedback). key section: 10.8 (`sec:meshdep:inner-cg`) is where the novel inner-cg pc result is written up. 10.4 covers the lmvm history-size fix. table 10.z (the tao solver zoo) is around line 1497
- `biblio.bib`: bibliography

### `preliminary_report/`
- `preliminary_report.tex`: 3-page report submitted 6/7/26. structure: intro / background / objectives / methods / results / work plan. includes footnotes to two firedrake demos
- `preliminary_report.bib`: 7 bibtex entries covering schwedes, kirby, firedrake, pyadjoint, hpuu, netgen, gunnel-herzog-sachs

### `notes/`
- `next_meeting_briefing.tex`: 11-page briefing for the meeting that set the four-item plan. explains why each step matters, where the ideas come from (schwedes + firedrake saddle-point demo), what the outcome was for items 1 to 3, and for item 4 what to run and how to read the output
- `next_meeting_briefing.pdf`: compiled version
- `progress_log.md`: this file

### `poster/`
- `poster.tex`: a1 portrait poster in `beamerposter` + `pgfplots`. two columns. currently uses the whole-domain random-refined mesh figures (reverted from interior for david's first look)
- `poster.pdf`: compiled output
- `gen_mesh_figure.py`: generates netgen or schwedes random-refined mesh figures at selected levels/ratios
- `gen_localised_mesh_figure.py`: generates corner-localised random-refined mesh figures (`[0, 0.4]^2`)
- `gen_interior_mesh_figure.py`: generates interior-localised random-refined mesh figures (`[0.3, 0.7]^2`)
- `gen_comparison_plot.py`: matplotlib log-log plot comparing netgen vs random-refined mesh sequences on the same axes
- `figures/`: the actual pdf/png files used by the poster, including `imperial_logo.png`, various mesh views, and comparison plots

### `code/meshdep/` (the main library code)
- `__init__.py`: package marker with docstring "code for mesh-dependence in pde-constrained optimisation"
- `meshes.py`: mesh constructors. `uniform_unit_square(n)`, `graded_unit_square_from_file(path)`, `graded_unit_square_tensor(h_ratio, n=32)` (fallback exp-stretch), `graded_unit_square(h_ratio, h_max=0.2)` (uses netgen directly)
- `optimisers.py`: three wrappers. `solve_with_tao(Jhat, ...)` for tao/lmvm through pyadjoint. `solve_with_scipy(Jhat)` for scipy l-bfgs-b in $\ell^2$. `solve_with_scipy_external_check(Jhat, test_riesz_map="L2", ...)` for scipy with convergence checked in a chosen hilbert-space norm rather than $\ell^2$. `solve_with_hilbert_lbfgs(Jhat, history=5, ...)` for the custom l-bfgs with every dot product taken in the hilbert inner product chosen on the control. this is the reference "flat row" implementation
- `preconditioners.py`: the riesz-map pcs. `RieszMapL2` and `RieszMapH1` are `AuxiliaryOperatorPC` subclasses. override `form()` to return the mass or helmholtz form. framework-native but silently fails for tao's inner ksp. `RieszMapPCContext` is a plain petsc python pc context. takes the function space $V$ explicitly at construction. assembles $M_h$ (or helmholtz), factorises via lu, applies on every call. this is what the tests actually use

### `code/meshdep/problems/`
- `__init__.py`: "forward solvers and optimal control wrappers for the poisson and burgers equations"
- `forward_poisson.py`: forward poisson solve with a manufactured solution, returns the $L^2$ error against `u_exact`
- `forward_poisson_byhand.py`: same problem, but plain python with p1 elements and reference-triangle basis functions. used as a sanity check that we understand the discretisation
- `poisson_control.py`: poisson optimal control problem
- `burgers.py`: 1d viscous burgers optimal control (defined here)
- `burgers_control.py`: firedrake + pyadjoint wrappers for the burgers control

### `code/experiments/` (top-level scripts)
- `poc_uniform.py`: proof of concept on uniform meshes
- `table_2_2.py`: the main panel (a) reproduction script. runs each (mesh, optimiser, riesz_map) combination for a panel and prints iteration counts
- `extended.py`: extends table 2.2 beyond $h_{\max}/h_{\min} = 128$

### `code/mesh_generation/`
- `generate_meshes.py`: script to produce the six netgen graded `.msh` files at target $r \in \{4, 8, 16, 32, 64, 128\}$. netgen's python package doesn't provide aarch64 binaries, so meshes are generated once on an x86_64 machine and stored as `.msh` files. the file header is patched from gmsh v2.0 to v2.2 so firedrake will read it
- `.msh` files: the six cached graded meshes. `graded_R4.msh`, `graded_R8.msh`, ..., `graded_R128.msh`

### `code/investigations/`
- `__init__.py`: package marker
- `random_refine.py`: schwedes sect 2.3.2.1 random rivara refinement scheme. `random_refined_mesh(level, seed=42, p_refine=0.35, n_initial=8)` returns `(mesh, realised_ratio, n_cells)`. also low-level helpers (`initial_uniform_arrays`, `bisect_refine`, `check_conforming`, `h_ratio`, `write_gmsh22`)
- `minimal_reproducer.py`: the minimal reproducer for the lmvm history-size bug from week of 2/6
- `original_failure_reproducer.py`: historical script preserved for the record

### `code/investigations/test/` (all the sweep and diagnostic scripts)

item-3-family (riesz pc on inner ksp):
- `test_riesz_pc.py`: first riesz pc test. confirmed the options-database route was silently ignored
- `test_riesz_pc_sweep.py`: headline item 3 script on netgen graded meshes. produces the 56, 147, 432, 807 (baseline) vs flat 15 (with pc) numbers used in 10.8 and on the poster
- `test_localised_riesz_pc_sweep.py`: item 3 on corner-localised random-refined meshes $[0, 0.4]^2$
- `test_interior_riesz_pc_sweep.py`: item 3 on interior random-refined meshes $[0.3, 0.7]^2$
- `test_random_localised.py`: parameter sweep confirming that domain-wide random refinement fundamentally can't reach high ratios, and showing that a sub-region restriction unlocks them

item-4-family (bqnls diagnostic):
- `test_bqnls_line_search.py`: bqnls with `tao_ls_monitor` on the netgen sweep at hist=5 and hist=100 (via the wrong option prefix). produced the wrong-in-retrospect identical counts
- `test_bqnls_h0_fix.py`: corrected sweep using pyadjoint's `parameters` dict, and the first tao_view probe. also tries scale_type overrides
- `test_bqnls_seed_install.py`: the three-route probe that achieved the causal proof of hypothesis c. routes 1 and 2 install $M_h$ as $J_0$ on bqnls via `Mat.setLMVMJ0`, flattening the row to 19, 18, 23, 18

lmvm history-size story:
- `test_lmvm_h0_rescaling.py`: demonstrates pyadjoint's taosolver disables l-bfgs initial-hessian rescaling in specific configurations. superseded by the history-size fix but still documents the earlier hypothesis

lmvm vs nls side-by-side:
- `test_poisson_lmvm_vs_nls.py`: on netgen meshes
- `test_poisson_lmvm_vs_nls_stretch.py`: on exp-stretched meshes
- `test_poisson_lmvm_vs_nls_random.py`: on schwedes random-refined meshes
- `test_burgers_lmvm_vs_nls.py`: burgers version
- `test_burgers_lmvm_vs_nls_stretch.py`: burgers on exp-stretch

tao solver zoo:
- `test_tao_solver_zoo.py`: every tao solver on poisson, netgen meshes. produces dissertation table 10.z
- `test_tao_solver_zoo_burgers.py`: same on 1d burgers (invalidated by the exp-stretch boundary-collapse issue)

h1 experiments:
- `test_h1_riesz.py`: try `riesz_map="H1"` on control as an alternative fix to the lmvm history problem

sanity checks:
- `test_norm_trajectory.py`: monitor both $L^2$ and $\ell^2$ gradient norms at every outer iteration
- `test_resolution.py`: plot the optimised solution at various grading levels to check that high-stretch meshes haven't turned the problem into a different one
- `test_burgers_lmvm.py`: does $H^1$ alone (no history hack) survive on burgers?
- `test_burgers_nls.py`: does nls misbehave on a nonlinear problem?
- `test_nls.py`: nls on netgen poisson control (baseline check)
- `test_poisson_monitor.py`: per-iteration monitor trace of nls and lmvm (david's line-search suspicion)

poster data sweeps:
- `run_random_sweep_for_poster.py`: full sweep on whole-domain random-refined meshes (levels 0, 2, 4, 6, 8, 10). scipy + hilbert + baseline nls + nls+pc
- `run_interior_sweep_for_poster.py`: same on interior random-refined meshes (levels 4, 8, 12, 14). interrupted at level 14 (scipy l-bfgs-b step)

visualisation:
- `visualise_netgen_meshes.py`: renders the six cached netgen meshes as pngs plus a 2x3 grid
- `visualise_random_refined_meshes.py`: renders the schwedes random-refined mesh sequence

### `code/test/` (correctness tests)
- `__init__.py`: package marker
- `test_01_forward_poisson_convergence.py`: poisson forward solver converges at rate 2 in $L^2$ with p1
- `test_02_poisson_taylor.py`: taylor test on the poisson-control reduced functional. confirms the adjoint gradient is correct: $|J(m + h\, dm) - J(m) - h\, dJ(m; dm)| = O(h^2)$
- `test_03_riesz_maps.py`: smoke test that `riesz_map` on control changes the gradient (rules out pyadjoint silently ignoring the keyword)
- `test_04_burgers_convergence.py`: burgers forward solver converges at expected rate
- `test_05_forward_poisson_byhand.py`: plain-python p1 poisson forward solver converges at rate 2

### `code/old_code/`
- `forward_poisson.py`, `forward_poisson_convergence.py`: historical versions kept for the record

---

## running technical themes

### mesh constructions actually used

| construction | file | reachable $h_{\max}/h_{\min}$ | notes |
|---|---|---|---|
| netgen graded | `mesh_generation/graded_R{r}.msh` | 5.9 to 194 | fine sub-square $(0, 0.4)^2$. deterministic. primary construction for 10.8 and table 2.2 reproduction |
| schwedes random-refined (whole domain) | `random_refine.py` | 1 to 5.7 (capped) | lepp chain caps the ratio. used for the current poster panel-(a) table |
| localised random (corner $[0, 0.4]^2$) | `test_localised_riesz_pc_sweep.py` | 4 to 128 | random within a corner sub-square. touches dirichlet edges |
| localised random (interior $[0.3, 0.7]^2$) | `test_interior_riesz_pc_sweep.py` | 4 to 128 | random within an interior sub-square. no dirichlet interaction. david's preferred |
| synthetic exp-stretched | `graded_unit_square_tensor` in `meshes.py` | any | boundary-collapse issue flagged by supervisor 27/5. not used since then |

### headline results across all constructions

| experiment | mesh construction | mesh-dependent row | fix row |
|---|---|---|---|
| panel (a) outer l-bfgs | netgen $r = 4, 16, 64$ | scipy $\ell^2$: 46, 195, 786 | hilbert $L^2$: 8, 7, 8 |
| panel (a) outer l-bfgs | random whole-domain L = 0..10 | scipy $\ell^2$: 13, 31, 35, 47, 47, 55 | hilbert $L^2$: 6, 8, 6, 6, 5, 4 |
| panel (a) outer l-bfgs | interior random L = 4, 8, 12 | scipy $\ell^2$: 71, 205, 701 | hilbert $L^2$: 6, 7, 8 |
| item 3 inner cg | netgen $r = 4, 16, 64, 128$ | baseline: 56, 147, 432, 807 | + riesz pc: flat at 15 |
| item 3 inner cg | corner random L = 4, 8, 12, 14 | baseline: 63, 186, 552, 973 | + riesz pc: flat at 14 |
| item 3 inner cg | interior random L = 4, 8, 12, 14 | baseline: 65, 197, 610, 1111 | + riesz pc: 13-14 |
| item 4 bqnls outer iters | netgen $r = 4, 16, 64, 128$ | default (hist=5, diagonal $J_0$): 33, 60, 88, 118 | with $M_h$ as $J_0$ at hist=100: 19, 18, 23, 18 |

### solver-family behaviour under grading (from table 10.z)

- newton-family (`nls`, `bnls`, `bntr`, `bntl`): outer count trivially small on poisson (quadratic). mesh-dependence hidden in the inner cg for `nls`. fixed by riesz pc. 10.8 novel result
- lmvm-family (`lmvm`): mesh-independent once history is bumped from default 5 to 100. 10.4
- bounded-qn-family (`bqnls`): mesh-dependent at defaults. two causes stacked: history too small + no $M_h$ seed on $J_0$. fixed today (7/7) via `Mat.setLMVMJ0`. item 4 conclusion
- cg (`cg`): mesh-dependent, budget-limited at high ratios. written up as-is in the dissertation

---

## open items (as of 14/7/26)

done since 7/7: poster submitted (13/7); pyadjoint `RieszMapPC` built + tested (matches the reference exactly on l2); h1 mesh-dependence investigated and resolved (regularisation-matching); "a problem where l2 fails but h1 wins" found (the h1-regularised test); $H^1$ riesz pc sweep done (it grows on the l2-reg problem, as it should).

- **pyadjoint riesz pc**: class built and tested. remaining: (a) settle the import-safety approach with josh (slack sent/pending), (b) move into `tao_solver.py` if agreed, (c) write the firedrake test in `tests/firedrake/adjoint/test_optimisation.py`, (d) open the two prs (pyadjoint + firedrake) and ping david + josh
- **h1 open question**: ask david exactly which h1 operator he means for the l2-reg problem. the plain h1 riesz map (M+K) is verified mesh-dependent (theory + same-optimum experiment + literature); a matching / sqrt(alpha)-weighted operator preconditioner (schoberl-zulehner type) might be mesh-robust for l2-reg and is untested
- **dissertation writing**: david wants at least half the remaining time on this. draft-with-holes to him by next week, before the remote meeting week of 27/7. middle sections still half-bullet-points. needs the item 4 (bqnls) write-up and now the h1 / regularisation-matching story
- burgers experiments still open. 2d burgers on the netgen graded meshes is the likely route (1d exp-stretch has the boundary-collapse pathology per supervisor 27/5)
- upstream report of the pyadjoint bqnls pc gap (github issue)
- correct the dissertation claim that history size alone causes bqnls mesh-dependence (it's history + missing $J_0$ seed together)
- random-refined meshes future-work claim (dissertation line 1867): now largely done via the interior-region variant

timeline: david away from ~8/8. remote meeting week of 27/7 (not week of 3/8). roughly 3 weeks of working time left. poster presentation 21/7 13:00.

---

## meeting log

| date | attendees | main outcomes |
|---|---|---|
| 24/3/26 | david | project scoped, reproduce schwedes ch 2 |
| 27/5/26 | david | rol-to-tao reframing (matt knepley discord). three-goal structure set |
| 9/6/26 | david | tao solver zoo built and reviewed |
| 23/6/26 | david | inner-cg idea. four-item plan set (riesz pc + bqnls diagnostic) |
| meeting (early 7/26) | david | preliminary report review, poster prep |
| 7/7/26 | david, josh hope-collins | poster feedback. pyadjoint riesz pc refactor scoped. josh joins for monday. dissertation writing prioritised |
| ~13/7/26 | david, josh hope-collins | riesz pc reviewed (gradJ rename, narrow type check, view method, move to tao_solver). pr plan set (fork firedrake, tests in test_optimisation.py, prs into both, ping on slack). import-safety flagged. h1 investigation direction. draft-with-holes due next week. david away ~8/8, remote week of 27/7 |
