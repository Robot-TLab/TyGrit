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

### POMDP

```{raw} html
<video controls width="100%" style="margin: 1em 0;">
  <source src="_static/videos/pomdp.mp4" type="video/mp4">
</video>
```

The standard POMDP graphical model assumes $T$ and $\Omega$ are **given** — the
transition and observation models are inputs to the solver. In our problem, both are
unknown and must be discovered through interaction.

We *can* fold the unknown parameters into the state:
$s^+ = (s, \theta_T, \theta_\Omega, \theta_S)$. This gives a Bayes-Adaptive POMDP.
But every belief update requires evaluating $\Omega(o \mid s', a)$ — the likelihood of a
specific image given a world state and camera pose. This is **the entire problem of
computer vision**. The POMDP doesn't solve perception; it buries it inside $\Omega$ and
assumes it's given.

Even if tractable, the optimal policy would rediscover structure we already know: observe
before reaching, maintain line-of-sight, replan after surprises. Intractable computation
to **rediscover known structure**.

### RL / VLAs

Visuomotor policies learn $a = \pi(o_{t-k}, \ldots, o_t)$. No model needed — sidesteps
the specification problem entirely.

But a reactive policy has **no concept of information sufficiency**. It cannot represent
"I don't know enough to act safely" or decide to stop and look before reaching.

Safety is statistical: the policy is safe in states it trained on. In novel
configurations, there is no mechanism to guarantee collision avoidance. **The policy
doesn't know what it doesn't know.**

And there is **no lookahead** — it cannot anticipate that a planned arm motion will
occlude the camera three steps from now.

### Model Predictive Control

MPC solves a finite-horizon optimization at each step. It assumes the state $x_t$ is
known, the dynamics $f$ are known, and the constraint set $g$ can be evaluated.

In our problem, none hold with certainty. MPC plans through a **known** feasibility set.
Ours is discovered through observation.

MPC has no concept of information: it cannot reason about what the robot can or cannot
see, and cannot take actions to improve future observability.

### What we need

A framework that:

- Plans over a **finite horizon** — the world is only locally known
- Treats **information as a first-class quantity** — not buried in belief, not ignored
- Handles **unknown, evolving models** — not given upfront
- Provides **verifiable safety** — not statistical
- Exploits **known task structure** — not rediscovering it through search

See {doc}`architecture` for how TyGrit addresses this.


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
