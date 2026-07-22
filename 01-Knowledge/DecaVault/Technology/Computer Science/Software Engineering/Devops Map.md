
## CI/CD Learning Roadmap (Developer-Focused)

### Phase 0 — Mental Model (Day 0–1)

**Goal:** Understand *what* CI/CD is doing before touching tools.

#### Concepts to Learn

* What happens from `git push` → production
* CI vs CD (clear separation)
* Pipeline = code (not configuration)
* Immutable artifacts
* Environments ≠ branches

#### Outcome

You can explain, end-to-end, how a commit becomes a running service.

---

### Phase 1 — Git & Triggers (Days 1–3)

**Goal:** Understand what triggers pipelines and why.

#### Learn

* Branching strategies:

  * Trunk-based
  * Feature branches
* PR-based pipelines
* Tag-based releases
* Commit SHA vs semantic versions
* Merge vs rebase impact on CI

#### Hands-on

* Create a repo
* Add:

  * PR trigger
  * Main-branch trigger
  * Tag trigger

#### Outcome

You can predict **when** and **why** a pipeline will run.

---

### Phase 2 — CI Fundamentals (Days 4–7)

**Goal:** Build a basic but real CI pipeline.

#### Learn

* Pipeline stages:

  * Checkout
  * Dependency install
  * Lint
  * Unit test
  * Build
* Job vs step
* Runners / agents
* Caching dependencies
* Fail-fast behavior

#### Hands-on

* Create a CI pipeline that:

  * Installs dependencies
  * Runs tests
  * Builds an artifact
  * Fails on test errors

#### Outcome

You can read and debug CI logs confidently.

---

### Phase 3 — Testing Strategy in CI (Days 8–10)

**Goal:** Know *what* to test *where*.

#### Learn

* Unit vs integration tests
* Test pyramid
* Test isolation
* Coverage reporting
* Flaky tests and retries

#### Hands-on

* Add:

  * Unit tests in CI
  * Coverage report
* Break a test intentionally and fix it

#### Outcome

You understand why a pipeline is red—and whether it *should* be.

---

### Phase 4 — Artifact & Versioning (Days 11–13)

**Goal:** Understand how builds are reused and promoted.

#### Learn

* Artifacts vs deployments
* Artifact immutability
* Docker images as artifacts
* Versioning:

  * SemVer
  * Commit SHA tagging
* Promotion (not rebuilding)

#### Hands-on

* Build a Docker image in CI
* Push to a registry
* Tag with commit SHA

#### Outcome

You understand why “rebuilding for prod” is a bad practice.

---

### Phase 5 — Configuration & Secrets (Days 14–16)

**Goal:** Avoid the #1 real-world CI/CD mistake.

#### Learn

* Config vs code
* Environment variables
* Secrets:

  * CI secrets
  * Vaults / cloud secret managers
* Masking and leakage risks
* Feature flags (conceptual)

#### Hands-on

* Inject env vars via pipeline
* Inject secrets securely
* Verify secrets are masked in logs

#### Outcome

You can deploy the same artifact to multiple environments safely.

---

### Phase 6 — CD & Deployment Basics (Days 17–20)

**Goal:** Understand how code reaches runtime.

#### Learn

* Deployment pipelines vs CI pipelines
* Manual vs automated deploys
* Health checks
* Rollbacks vs roll-forwards
* Deployment failures

#### Hands-on

* Deploy to a non-prod environment
* Add a health check
* Simulate a failed deployment

#### Outcome

You can reason about production safety.

---

### Phase 7 — Deployment Strategies (Days 21–23)

**Goal:** Learn how downtime is avoided.

#### Learn

* Rolling deployments
* Blue–green
* Canary releases
* Traffic shifting
* Zero-downtime assumptions

#### Hands-on

* Implement rolling deployment
* Compare with blue–green conceptually

#### Outcome

You understand *why* deployment strategies matter.

---

### Phase 8 — Docker Deep Dive (Days 24–26)

**Goal:** Eliminate container-related CI/CD failures.

#### Learn

* Dockerfile best practices
* Multi-stage builds
* Layer caching
* Image size optimization
* Security basics

#### Hands-on

* Optimize a Dockerfile
* Reduce image size
* Add `.dockerignore`

#### Outcome

You can debug Docker-related pipeline failures.

---

### Phase 9 — Kubernetes Awareness (Days 27–30)

**Goal:** Be effective even if infra is owned by another team.

#### Learn

* Pod, Deployment, Service
* ConfigMap vs Secret
* Resource limits
* Readiness vs liveness
* Rollouts and rollbacks

#### Hands-on

* Deploy a service
* Trigger a rollout
* Debug a CrashLoopBackOff

#### Outcome

You are no longer blocked by “Kubernetes issues.”

---

### Phase 10 — Observability & Post-Deploy (Days 31–33)

**Goal:** Understand whether a deployment *actually worked*.

#### Learn

* Logs, metrics, alerts
* Deployment validation
* Error budgets
* Canary analysis (conceptual)

#### Hands-on

* Add logging
* Validate after deployment
* Monitor basic metrics

#### Outcome

You can validate deployments beyond “pipeline passed.”

---

### Phase 11 — Security in CI/CD (Days 34–36)

**Goal:** Avoid production and compliance incidents.

#### Learn

* Least-privilege CI roles
* Secret exposure risks
* Dependency scanning
* Image scanning
* Supply-chain risks

#### Hands-on

* Add dependency scanning
* Review CI permissions

#### Outcome

You write pipelines that security teams trust.

---

### Phase 12 — Failure Mastery (Days 37–40)

**Goal:** Become senior-level effective.

#### Learn

* Pipeline failure patterns
* Partial deployments
* Rollback failures
* Debugging production pipelines
* Incident response basics

#### Hands-on

* Break:

  * Build
  * Tests
  * Deployment
* Recover from each

#### Outcome

You are trusted to fix broken pipelines.

---

### What “CI/CD Proficient Dev” Means

You are CI/CD-proficient if you can:

* Predict pipeline behavior
* Debug failures quickly
* Modify pipelines safely
* Understand production impact
* Communicate clearly with DevOps/SRE

