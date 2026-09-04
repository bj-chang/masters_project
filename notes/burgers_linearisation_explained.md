# From the step residual (66) to the nudged expansion (73) — term by term

Notes on §8.2 of the dissertation. Current equation numbers:

| number | what it is |
|---|---|
| (66) | the step residual $F^{n+1}$ — the thing being differentiated |
| (72) | the convection linearisation — a tool prepared in advance |
| (73) | the nudged residual, expanded and collected by powers of $\varepsilon$ |
| (74) | the derivative $\partial_{u^{n+1}}F^{n+1}(w,v)$ — the $\varepsilon$-coefficient of (73) |

## The starting point: equation (66)

$$F^{n+1}(u^{n+1}, u^n, m^{n+1};\,v) = \int_\Omega \underbrace{\frac{u^{n+1}-u^n}{\Delta t}\,v}_{\text{(a) time}} + \underbrace{u^{n+1}\,\partial_x u^{n+1}\,v}_{\text{(b) convection}} + \underbrace{\nu\,\partial_x u^{n+1}\,\partial_x v}_{\text{(c) diffusion}} - \underbrace{m^{n+1}\,v}_{\text{(d) source}}\,\mathrm{d}x$$

The move that produces (73): **substitute $u^{n+1} + \varepsilon w$ for $u^{n+1}$, in every term, and expand.** Each term of (66) splits into an *original part* (no $\varepsilon$), an *$\varepsilon$-part* (the linear response), and possibly an $\varepsilon^2$ leftover. Nothing else happens — (73) is just (66) after this substitution, with the pieces sorted by powers of $\varepsilon$.

## Term (a), time: $\frac{u^{n+1}-u^n}{\Delta t}v$

Substitute and split the fraction:

$$\frac{(u^{n+1}+\varepsilon w)-u^n}{\Delta t}\,v = \underbrace{\frac{u^{n+1}-u^n}{\Delta t}\,v}_{\text{original}} + \varepsilon\,\underbrace{\frac{w\,v}{\Delta t}}_{\varepsilon\text{-part}}$$

$u^{n+1}$ appears once and linearly, so the split is exact — no $\varepsilon^2$ left over. The $-u^n$ is the old state; it is not being nudged, so it stays inside the original part.

## Term (b), convection: $u^{n+1}\,\partial_x u^{n+1}\,v$ — this is where (72) is used

Substituting gives $(u^{n+1}+\varepsilon w)\,\partial_x(u^{n+1}+\varepsilon w)\,v$, a product of two nudged brackets. Equation (72), applied with $u = u^{n+1}$ and $\delta u = w$, expands exactly this product:

$$(u^{n+1}+\varepsilon w)\,\partial_x(u^{n+1}+\varepsilon w) = \underbrace{u^{n+1}\partial_x u^{n+1}}_{\text{original}} + \varepsilon\underbrace{\bigl(w\,\partial_x u^{n+1} + u^{n+1}\partial_x w\bigr)}_{\varepsilon\text{-part (product rule)}} + \underbrace{\varepsilon^2\, w\,\partial_x w}_{\text{the } \mathcal{O}(\varepsilon^2)}$$

then everything is multiplied by the waiting $v$. Two terms in the $\varepsilon$-part because the nudge hits the two copies of $u^{n+1}$ one at a time; the $\varepsilon^2$ piece is both nudges at once.

## Term (c), diffusion: $\nu\,\partial_x u^{n+1}\,\partial_x v$

$$\nu\,\partial_x(u^{n+1}+\varepsilon w)\,\partial_x v = \underbrace{\nu\,\partial_x u^{n+1}\,\partial_x v}_{\text{original}} + \varepsilon\,\underbrace{\nu\,\partial_x w\,\partial_x v}_{\varepsilon\text{-part}}$$

Again once-and-linearly, so an exact split: the $\varepsilon$-part is the same term with $w$ standing where $u^{n+1}$ stood.

## Term (d), source: $-m^{n+1}v$

Contains no $u^{n+1}$ at all, so the substitution changes nothing:

$$-m^{n+1}v \;\longrightarrow\; \underbrace{-m^{n+1}v}_{\text{original}} \;+\; \varepsilon\cdot 0$$

It contributes only to the original part, and nothing to the $\varepsilon$-part.

## Reassembling: why (73) looks the way it does

Now add the four expansions back up under the integral and sort by powers of $\varepsilon$:

- **All four original parts together are precisely (66) again** — the un-nudged residual. That is why the right-hand side of (73) begins with $F^{n+1}(u^{n+1}, u^n, m^{n+1};\,v)$: it is not a new object, just terms (a)–(d) regrouping themselves.
- **The three $\varepsilon$-parts collect into the $\varepsilon\int_\Omega(\cdots)\mathrm{d}x$ bracket** of (73): $\frac{wv}{\Delta t}$ from (a), $(w\,\partial_x u^{n+1} + u^{n+1}\partial_x w)v$ from (b), $\nu\,\partial_x w\,\partial_x v$ from (c). Term (d) is absent because its response was zero.
- **The single $\varepsilon^2$ piece** (from the convection term) is bundled into the $\mathcal{O}(\varepsilon^2)$.

So (73) reads: *nudged residual = original residual + $\varepsilon\,\times$(sum of linear responses) + tiny quadratic leftover.*

## The last step: (73) to (74)

The difference quotient $\frac{F^{n+1}(u^{n+1}+\varepsilon w,\ldots) - F^{n+1}(u^{n+1},\ldots)}{\varepsilon}$ cancels the original part, leaving the $\varepsilon$-bracket plus $\mathcal{O}(\varepsilon)$; letting $\varepsilon \to 0$ kills the leftover. What survives — the bracket alone — is (74), the partial derivative $\partial_{u^{n+1}}F^{n+1}(w,v)$. Exactly the Poisson §5.2 procedure, applied to a residual with one nonlinear term.

## One-line summary

Each term of (66) splits under the nudge into "itself + $\varepsilon\,\times$ response": the responses are $\frac{wv}{\Delta t}$, the product-rule pair from (72), and $\nu\,\partial_x w\,\partial_x v$, with the source term responding not at all; the originals reassemble into (66), the responses form (73)'s $\varepsilon$-bracket, and that bracket is (74).
