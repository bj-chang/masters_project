# pyadjoint `TAOSolver` disables L-BFGS initial-Hessian rescaling

**Summary.** When `pyadjoint.optimization.tao_solver.TAOSolver` is used with an
LMVM/BLMVM TAO type, it installs the Riesz map `M⁻¹` as a **fixed** initial
Hessian via `tao.setLMVMH0(...)`. With a user-supplied `J0` set this way,
PETSc/TAO does not apply its automatic initial-Hessian scaling (the standard
L-BFGS rescaling `γ_k = ⟨s,y⟩/⟨y,y⟩`). The rescaling is therefore silently
switched off. On uniform meshes the fixed `M⁻¹` scale is close to optimal and
this is harmless, but on strongly graded meshes the fixed scale is far from
optimal and the iteration count grows by roughly **100×** — a mesh-dependence
that the correct (rescaled) L-BFGS does not have.

Environment: pyadjoint 2025.10.1, petsc4py 3.24.0, Python 3.12.3.

---

## Background

For the function-space optimisation in Schwedes et al. (2017), the control is
measured in `L²`, i.e. the gradient is the Riesz representation `M⁻¹ ∇J` where
`M` is the mass matrix. `TAOSolver` builds this correctly:

- `tao.setGradientNorm(M⁻¹)` so the stopping test uses the `L²` gradient norm;
- `tao.setLMVMH0(H0)` where `H0` applies `M⁻¹`, as the initial Hessian.

Standard L-BFGS additionally **rescales** that initial Hessian every step by the
scalar `γ_k = ⟨s,y⟩/⟨y,y⟩` (Nocedal & Wright, §7.2). This adjusts the *overall
magnitude* of the initial guess to the curvature actually observed. PETSc/TAO
implements this as the LMVM `scalar`/`diagonal` scale type, which is on by
default — *unless* the user supplies `J0`.

## Where it happens

`pyadjoint/optimization/tao_solver.py` (master:
<https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/optimization/tao_solver.py>;
installed 2025.10.1 line numbers below):

```python
# line 702-703
Minv_mat = RieszMapMat(rf.controls, comm=comm)
tao.setGradientNorm(Minv_mat)
...
# line 728
if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM}:
    ...
    # line 745-748: B_0_matrix applies M^{-1}
    B_0_matrix = PETSc.Mat().createPython(((n, N), (n, N)), InitialHessian(), comm=comm)
    ...
    # line 755-759
    tao.setLMVMH0(B_0_matrix)
    ksp = tao.getLMVMH0KSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
    ksp.setPC(B_0_matrix_pc)
```

`setLMVMH0` supplies a fixed `J0`; the PREONLY/`max_it=1` KSP applies it exactly
once per step. Once this `J0` is set, the LMVM scaling does not act on it.

## Proof

Script: `code/scripts/tao_rescaling_proof.py`. Problem: Poisson distributed
control on the unit square, `min ½‖u−d‖²_{L²} + ½α‖m‖²_{L²}` s.t. `−Δu=m`,
`α=1e-4`, P1 elements, LU forward solve, `tao_gatol=1e-7`. Canonical setup:
`Control(m)` (default Riesz map is `L²`) + `TAOSolver`, `tao_type=lmvm`.

### Test 1 — nothing looks wrong on uniform meshes

```
          mesh  iterations
  uniform N=32           4
  uniform N=64           4
 uniform N=128           3
```

Flat under refinement: mesh-independent, as expected.

### Test 2 — the standard scaling knob has no effect

Same problem on the graded mesh `graded_R16` (element-size ratio ≈ 24×),
sweeping the LMVM scale type:

```
    scale_type  iterations
          none         670
        scalar         670
      diagonal         670
```

`scalar` is the standard rescaling. It changes nothing, because the fixed `J0`
set by `setLMVMH0` overrides it. So the rescaling cannot be re-enabled by the
obvious option.

### Test 3 — decisive: the initial-Hessian scale is fixed, not rescaled

Same graded mesh; reproduce pyadjoint's `H0 = M⁻¹` exactly, then multiply it by
a constant `c` and sweep `c`:

```
             c  iterations
    none (c=1)         670     # untouched pyadjoint setup
         1e+00         670     # our faithful reproduction of it
         1e+01          79
         1e+02          26
         1e+03          12
         1e+04           7
```

**A rescaling optimiser is invariant to a constant scale on `H0`** — `γ_k` would
divide any constant back out, so every row would read ≈ 670. The count instead
collapses from 670 to 7 as `c` grows. This proves TAO is using the fixed scale
it was handed, i.e. it is **not** rescaling. (The `c=1` row matching the
baseline confirms the reproduction is faithful, so the sweep is trustworthy.)

A single constant `c ≈ 10⁴` reaches 7 iterations — matching a textbook rescaled
L-BFGS — and the optimal `c` tracks the (mesh-independent) curvature scale, so
this is a genuine fixed-scale artefact, not a per-mesh coincidence.

## Impact

| graded R16 (ratio ≈ 24×) | iterations |
|---|---|
| canonical `TAOSolver` (fixed `H0 = M⁻¹`) | 670 |
| same, but `H0` rescaled (constant `c ≈ 10⁴`) | 7 |
| reference: hand-written rescaled `L²` L-BFGS | 7 |

The mesh-dependence is entirely attributable to the missing rescaling, not to
the inner product (which is correct) nor the mesh (a rescaled method converges
in 7 on the same mesh).

## Suggested fixes

1. **Let TAO keep rescaling.** Provide the Riesz map without a fixed `J0` that
   disables scaling — e.g. via a change of variables `x = D m` (with
   `Dᵀ D = M`, exact or lumped) so plain LMVM in `ℓ²` *is* `L²` and TAO's own
   scaling applies. Verified to restore mesh-independence with stock TAO
   (`code/scripts/tao_rescaling_fix.py`: ~6–8 iters uniform, ~18–20 graded with
   lumped `M`).
2. **Apply scaling on top of the user `J0`.** If PETSc/TAO supports (or could
   support) `γ_k`-scaling a user-supplied `J0`, expose it; `TAOSolver` would
   then get rescaling for free.
3. **At minimum, document it.** `TAOSolver` silently turns off a standard L-BFGS
   feature whenever it sets the Riesz metric; users on non-uniform meshes should
   know.

## Reproduce

```
python code/scripts/tao_rescaling_proof.py     # the three tests above
python code/scripts/tao_rescaling_fix.py        # change-of-variables fix
```
