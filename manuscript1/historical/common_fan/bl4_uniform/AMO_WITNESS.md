# AMO common-fan witness: proof contract

This note isolates the mathematical contract checked by the independent
verifier. It is intentionally shorter than the generator code.

## 1. Coordinate chart

Let `v1,...,v5` be the vertices of a regular width-one 4-simplex centered at
its barycenter `G=0`, with

```text
|vi|^2 = 2/5,    vi.vj = -1/10  (i != j),
v5 = -(v1+v2+v3+v4).
```

A coefficient vector `a in Q^4` represents `sum a_i v_i`. Its metric is

```text
|a|^2 = (5*sum(a_i^2) - (sum a_i)^2)/10.
```

The Euclidean coordinate Jacobian is

```text
J = sqrt(5)/20.
```

Every permutation of the five simplex vertices acts by an integer `4 x 4`
matrix preserving this quadratic form.

The source paper uses an unnormalized regular simplex of edge length
`WIDTH = 2*z1`.  The verifier separately audits, with rational intervals, that
the four centered source basis vectors have Gram matrix exactly enclosed by
`WIDTH^2` times the width-one Gram matrix above.  This is the normalization
bridge from the published source coordinates to the coefficient chart; it is
not inferred from floating geometry.

## 2. Exact rational inball

The direct AMO body has the full regular-simplex symmetry and width one. The
minimum enclosing ball has a unique center, hence its center is the simplex
barycenter. Jung's theorem in dimension four gives circumradius at most

```text
sqrt(2/5).
```

For a width-one body centered at that point,

```text
h(u) + h(-u) = 1,
h(-u) <= sqrt(2/5),
```

so

```text
B(G, 1-sqrt(2/5)) subset K.
```

In particular the verifier may use the smaller exact rational ball

```text
B(G, 1/3) subset K.
```

## 3. Boundary source

The AMO source theorem constructs the boundary from three representative
chart types, transported by simplex symmetries:

- pair wedge;
- triple wedge;
- spherical 3-cap.

A witness names one of these types, a simplex permutation, and a rational
parameter point. The verifier evaluates the corresponding algebraic formula
with rational intervals and obtains a box containing a genuine source-proved
boundary point `x`.

Let `c` be the rational midpoint of that box and `rho` a rational upper bound
for its Euclidean radius. Then

```text
|x-c| <= rho.
```

The generator's floating optimizer is irrelevant once this witness is stored.

## 4. Inward-shrink implication

For rational `0 <= lambda <= 1`, convexity gives

```text
B(lambda*x, (1-lambda)/3)
    subset conv({x} union B(G,1/3))
    subset K.
```

Therefore a rational fan endpoint `p` is certified if

```text
|p-lambda*c| + lambda*rho <= (1-lambda)/3.
```

The verifier first checks that the right-hand slack is nonnegative and then
compares squared quantities in exact rational arithmetic.

## 5. Cap component guard

The homogeneous cone quadratic for the cap can have a remote second positive
component. The verifier does not accept `Q>=0` without a component argument.

In the gnomonic guard `0 <= r <= 1`, it audits the reduced form

```text
Q(r,z) = 1/8 + L*z*r + (A + 3*z^2/8)*r^2
```

by exact rational interval branch-and-bound. The present audit closes in 55
boxes and proves that every root in the guard is a down-crossing. Since
`Q(0,z)=1/8>0`, the accepted inequalities select precisely the component
containing `r=0`.

## 6. What is and is not proved

A verified AMO endpoint proves

```text
p in K
```

for that rational fan ray. Repeating this for every ray and checking the exact
fan subdivision gives an inscribed rational simplicial set. The determinant
sum is then an exact lower bound.

A shared ray cache may reuse the same certified endpoint across V1 levels or
V2 refinements.  The cache is generator-only: its schema and radial denominator
are preflighted, and every cache hit is passed through the exact endpoint
verifier again in the current fan before it is accepted.

The following are **not** proof steps:

- nearest-neighbour seed selection;
- floating least squares;
- QMC;
- floating Qhull;
- a visually plausible patch choice.

The common certificate becomes a proof of the desired comparison only when
the independent verifier returns

```text
strict_threshold_crossed = true
```

for `4437/20000`.
