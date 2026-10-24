# The paper's figures from the recorded results (seconds), or the results themselves (hours).
PY := uv run python
# one single-threaded JAX process per job, as the recorded runs
SINGLE := XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
SYSTEMS := duffing_accel fhn_rough lorenz96_da
# parallel jobs of the long runs
JOBS ?= 13

.PHONY: figures test results results-sigma-x-sweep results-failure-directions results-force-information \
	results-objectives results-robustness results-failures

# The figures `main.tex` includes, from the recorded results.
figures:
	$(PY) experiments/failure_directions.py --figure failure_directions_pi_no_ode_only
	$(PY) experiments/force_information.py --figure force_information_duffing force_information_extended_failures
	$(PY) experiments/objectives.py --figure comparison
	$(PY) experiments/failures.py --figure failures_by_method robustness_examples
	$(PY) experiments/robustness.py --figure robustness
	$(PY) experiments/sigma_x_sweep.py --figure sigma_x_sweep_all

test:
	uv run pytest

# --- recompute, hours in all. The targets are independent except where a prerequisite is named.
results: results-sigma-x-sweep results-failure-directions results-force-information results-objectives \
	results-robustness results-failures

# minutes (Duffing) to an hour (FitzHugh-Nagumo, Lorenz-96) per system
results-sigma-x-sweep:
	$(PY) experiments/sigma_x_sweep.py duffing fhn lorenz96

# ~20 s; the three diffusions sit around the Duffing maximiser of I(U; Y | Z) that the force information records
results-failure-directions: results-force-information
	$(PY) experiments/failure_directions.py duffing

# tens of minutes per system
results-force-information:
	$(PY) experiments/force_information.py duffing fhn lorenz96 fhn_rough_s0 duffing_accel_s10 lorenz96_da_s9

# The sigma_x^2 objectives over 20 draws per system at K0: one job per (system, draw, method), JOBS at a time
# (300 jobs, about an hour with 13). Fails if any job fails.
COMPARED := obs mi di fenrir_ieks classical_rk4
results-objectives:
	mkdir -p logs
	for b in $(SYSTEMS); do for d in $$(seq 0 19); do for m in $(COMPARED); do \
	  echo $$b $$d $$m; done; done; done | \
	xargs -P $(JOBS) -L 1 sh -c '$(SINGLE) $(PY) experiments/objectives.py $$0 --draw $$1 --methods $$2 \
	  > logs/objectives_$$0_$$1_$$2.log 2>&1'

# The objectives beyond the benchmark: forces not drawn from the prior on the benchmark systems, and the SIR model
# with a multiplicative force; one job per (system, force kind, draw, method), JOBS at a time (550 jobs, about an
# hour with 13).
results-robustness:
	mkdir -p logs
	uv run python -c "import sys; sys.path.insert(0, 'experiments'); import robustness as R; \
	  from lfcal.paper.benchmark import COMPARED; [print(n, k, d, m) for n, k, d in R.datasets() for m in COMPARED]" | \
	xargs -P $(JOBS) -L 1 sh -c '$(SINGLE) $(PY) experiments/robustness.py $$0 --kind $$1 --draw $$2 --methods $$3 \
	  > logs/robustness_$$0_$$1_$$2_$$3.log 2>&1'

# The posteriors of the wrong fits that the failure figures show, JOBS at a time (minutes per fit).
results-failures: results-objectives results-robustness
	$(PY) experiments/failures.py --compute --jobs $(JOBS)
