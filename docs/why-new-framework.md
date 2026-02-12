# Why a New Framework

A robot enters an unfamiliar room, finds an object, and picks it up. This requires
perception, planning, and control — in a world the robot has never seen, with objects
it has never touched, under physical constraints it cannot violate.

Every standard framework assumes some part of this problem away. This page shows
exactly what is missing and why it matters.


## Start from what everyone knows

A POMDP is the standard framework for decision-making under partial observability.
It is defined by seven components, all **given before execution begins**:

$$
\langle\, S,\; A,\; O,\; T,\; \Omega,\; R,\; \gamma \,\rangle
$$

| Symbol | Meaning |
|--------|---------|
| $S$ | State space — all possible world configurations |
| $A$ | Action space — all robot commands |
| $O$ | Observation space — all sensor readings |
| $T(s' \mid s, a)$ | Transition model — how the world evolves |
| $\Omega(o \mid s', a)$ | Observation model — what the sensor returns |
| $R(s, a)$ | Reward — task objective |
| $\gamma$ | Discount factor |

The only runtime unknown is $s_t$: the current state. Everything else — the state
space, the dynamics, the sensor model — is fully specified upfront.

The word "partial" in POMDP refers exclusively to the state:
**you know how the world works; you just don't know where things are right now.**


## Now look at our problem

In mobile manipulation in unknown environments, the situation is different.
Mark what is known vs. unknown at execution time:

```{raw} html
<div class="tuple-compare">
  <div>
    <span class="label">POMDP:</span>
    &#10216;
    <span class="known">S</span>,
    <span class="known">A</span>,
    <span class="known">O</span>,
    <span class="known">T</span>,
    <span class="known">&Omega;</span>,
    <span class="known">R</span>,
    <span class="known">&gamma;</span>
    &#10217;
    <span class="note">unknown: s<sub>t</sub></span>
  </div>
  <div>
    <span class="label">Our problem:</span>
    &#10216;
    <span class="unknown">?</span>,
    <span class="known">A</span>,
    <span class="known">O</span>,
    <span class="unknown">?</span>,
    <span class="unknown">?</span>,
    <span class="known">R</span>,
    <span class="known">&gamma;</span>
    &#10217;
    <span class="note">unknown: s<sub>t</sub>, S, T, &Omega;</span>
  </div>
</div>
```

Four unknowns instead of one:

- **$S$ is unknown** — how many objects exist, what types, what degrees of freedom.
  The state space itself is discovered through observation. An occluded object, once
  revealed, expands $S$.
- **$T$ is unknown** — object masses, friction coefficients, contact dynamics. Will
  this object slide when pushed? You don't know until you try.
- **$\Omega$ is unknown** — object geometry and appearance from unvisited viewpoints
  don't exist in the model yet. $\Omega$ is constructed as the robot observes.

A POMDP solver takes $\langle S, A, O, T, \Omega, R, \gamma \rangle$ as **input**.
If three of those components are missing, the solver cannot be invoked.

**The problem specification itself is incomplete, and must be completed
through interaction — under safety constraints — while solving the problem.**


## What about existing approaches?

```{raw} html
<div class="framework-cards">

  <div class="card">
    <h4>POMDP</h4>
    <p>We <em>can</em> fold the unknown model parameters into the state:
    <em>s<sup>+</sup> = (s, &theta;<sub>T</sub>, &theta;<sub>&Omega;</sub>,
    &theta;<sub>S</sub>)</em>. This gives a Bayes-Adaptive POMDP. Well-defined in
    principle.</p>
    <p>But every belief update requires evaluating &Omega;(o | s', a) — the likelihood
    of a specific image given a world state and camera pose. Inverting this is
    <strong>the entire problem of computer vision</strong>. The POMDP doesn't solve
    perception; it buries it inside &Omega; and assumes it's given.</p>
    <p>The value function V(b) is non-convex, discontinuous at feasibility boundaries,
    and task-phase-dependent. This breaks all known approximation algorithms.</p>
    <p>And if we <em>could</em> solve it, the optimal policy would rediscover structure
    we already know: observe before reaching, maintain line-of-sight, replan after
    surprises. Intractable computation to <strong>rediscover known structure</strong>.</p>
    <div class="verdict fail">Structurally intractable. Rediscovers known structure.</div>
  </div>

  <div class="card">
    <h4>RL / VLAs</h4>
    <p>Visuomotor policies learn <em>a = &pi;(o<sub>t-k</sub>, &hellip;,
    o<sub>t</sub>)</em>. No model needed — sidesteps the specification problem entirely.</p>
    <p>But a reactive policy has <strong>no concept of information sufficiency</strong>.
    It cannot represent "I don't know enough to act safely" or decide to stop and look
    before reaching.</p>
    <p>Safety is statistical: the policy is safe in states it trained on. In novel
    configurations, there is no mechanism to guarantee collision avoidance. <strong>The
    policy doesn't know what it doesn't know.</strong></p>
    <p>And there is <strong>no lookahead</strong> — it cannot anticipate that a planned
    arm motion will occlude the camera three steps from now.</p>
    <div class="verdict fail">No information reasoning. No safety guarantees. No lookahead.</div>
  </div>

  <div class="card">
    <h4>Model Predictive Control</h4>
    <p>MPC solves a finite-horizon optimization at each step. It assumes the state
    <em>x<sub>t</sub></em> is known, the dynamics <em>f</em> are known, and the
    constraint set <em>g</em> can be evaluated.</p>
    <p>In our problem, none hold with certainty. MPC plans through a
    <strong>known</strong> feasibility set. Ours is discovered through observation.</p>
    <p>MPC has no concept of information: it cannot reason about what the robot can or
    cannot see, and cannot take actions to improve future observability.</p>
    <div class="verdict fail">Assumes full observability. No information model.</div>
  </div>

  <div class="card">
    <h4>What we need</h4>
    <p>A framework that:</p>
    <p>&bull;&ensp;Plans over a <strong>finite horizon</strong> — the world is only locally known</p>
    <p>&bull;&ensp;Treats <strong>information as a first-class quantity</strong> — not buried in belief, not ignored</p>
    <p>&bull;&ensp;Handles <strong>unknown, evolving models</strong> — not given upfront</p>
    <p>&bull;&ensp;Provides <strong>verifiable safety</strong> — not statistical</p>
    <p>&bull;&ensp;Exploits <strong>known task structure</strong> — not rediscovering it through search</p>
    <div class="verdict" style="color: var(--color-foreground-primary);">
      &#8594; See <a href="architecture.html">Architecture</a> for how TyGrit addresses this.
    </div>
  </div>

</div>
```


## The core difficulty

The failures above share a common root. The fundamental difficulty is not partial
observability, not search space size, not multi-objective optimization.

It is a **circular dependency**:

```{raw} html
<div class="circular-dep">
  safe action &ensp;&larr;&ensp; information &ensp;&larr;&ensp; observation
  &ensp;&larr;&ensp; positioning the body &ensp;&larr;&ensp; safe action
  &ensp;&larr;&ensp; &hellip;
</div>
```

To act safely, the robot needs information. To get information, it must observe. To
observe, it must position its body (the camera is on the robot). To position its body,
it must act safely.

In standard planning, feasibility is **exogenous** — the free space exists independent
of the planner. Here, feasibility is **endogenous** — whether a plan is safe depends on
what the robot can observe, which depends on the robot's configuration, which depends on
the plan.

**The plan must verify itself.**

| | Feasibility set | Information | Action-observation coupling |
|---|---|---|---|
| **MPC** | known, exogenous | not modeled | none |
| **POMDP** | uncertain, but model given | modeled in belief | model assumed known |
| **RL / VLA** | not explicit | not explicit | learned implicitly |
| **Our problem** | **unknown, endogenous** | **discovered online** | **coupled through embodiment** |

This self-referential structure — the plan depends on the information the plan
generates — is what makes the problem different from those addressed by existing
frameworks, and what motivates the visibility-constrained receding horizon approach
described in {doc}`architecture`.
