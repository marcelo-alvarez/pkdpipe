# Workflow Wisdom

## Environment Setup for Python Testing
ALWAYS source the environment before testing Python imports or running Python code directly:
- CORRECT: `source ./load_env.sh && python -c "from pkdpipe.ngp_gridder import NGPGridder; print('NGP import OK')"`
- WRONG: `python -c "from pkdpipe.ngp_gridder import NGPGridder; print('NGP import OK')"` (missing environment)

This is ONLY needed for direct Python commands. `./run_examples.sh` automatically sources the environment internally.

## run_examples.sh Debug Flag
Use `--debug-synthetic` NOT `--debug`. The script has a bug passing SLURM args to Python causing "unrecognized arguments" errors.

## run_examples.sh Redirection Issue
When calling run_examples.sh, DO NOT use "2>&1" redirection. It causes the "2" to be incorrectly parsed as a script argument.
- WRONG: `./run_examples.sh --variant lcdm --ngrid 256 2>&1 | tee log`
- CORRECT: `./run_examples.sh --variant lcdm --ngrid 256 | tee log`

## 🚨 CRITICAL: JAX+MPI HANG DETECTION PROTOCOL 🚨
**FUNDAMENTAL TRUTH**: JAX+MPI parallel workflows created by Claude Code are prone to hanging due to architectural complexities that Claude struggles with. When a job appears stuck, IT IS HUNG - DO NOT WASTE TIME WITH WORKAROUNDS.

**HANG INDICATORS**:
- Job runs >5 minutes without new log output after gridding 
- FFT phase takes longer than gridding phase (should be similar or faster)
- Processes stuck after "gridding complete" messages
- Any pause >2 minutes in a phase that normally completes in seconds

**FORBIDDEN RESPONSES TO HANGS**:
❌ DO NOT extend time limits - hangs don't resolve with more time
❌ DO NOT reduce grid size - a 64³ or 256³ grid is already tiny
❌ DO NOT reduce particle counts - this won't fix synchronization issues
❌ DO NOT try "one more time with different parameters"
❌ DO NOT assume it's "just slow" - these operations should be fast

**CORRECT RESPONSE TO HANGS**:
1. IMMEDIATELY cancel the job - don't wait
2. Recognize this is a JAX+MPI synchronization bug
3. Focus on the ROOT CAUSE: distributed FFT or MPI communication deadlock
4. Debug the actual parallel coordination issue, not symptoms
5. If multiple attempts hang at same place, THE CODE IS BROKEN

**Why This Happens**: Claude Code often creates subtle bugs in distributed JAX workflows:
- Mismatched collective operations across ranks
- Incorrect device mesh initialization 
- Race conditions in distributed FFT setup
- Improper MPI barrier placement
- JAX compilation vs execution phase conflicts

**REMEMBER**: If a job hangs once, it will hang again with the same code. Fix the bug, don't work around it.

## 🚨 ABSOLUTE REQUIREMENT: JAX DISTRIBUTED FFT - NO FALLBACKS 🚨

**CRITICAL MANDATE**: The power spectrum calculation MUST use JAX distributed FFT. This is non-negotiable.

**ABSOLUTELY FORBIDDEN**:
- ❌ Numpy FFT fallbacks for "temporary" workarounds
- ❌ Single-node FFT fallbacks when distributed hangs
- ❌ CPU-only fallbacks when GPU+MPI fails
- ❌ Any mechanism that bypasses JAX distributed processing
- ❌ "Just get it working" approaches that avoid the real problem

**THE ONLY ACCEPTABLE SOLUTION**: Fix the JAX+MPI synchronization bug properly.

**Why No Fallbacks**:
- JAX distributed FFT is the core requirement for large-scale cosmological analysis
- Fallbacks mask the real distributed computing bugs that need fixing
- The codebase must work reliably in production distributed environments
- N=5 outlier investigation requires proper distributed FFT behavior

## NO FALLBACK MECHANISMS IN DISTRIBUTED CODE
CRITICAL: When fixing distributed computing issues, DO NOT implement fallback mechanisms or timeouts that switch to single-node mode.

**WRONG APPROACH**:
- Timeout handlers that fall back to CPU-only mode
- Exception handlers that disable distributed processing
- "Graceful degradation" to single-node execution
- Any mechanism that hides or works around distributed failures

**CORRECT APPROACH**:
- Fix the actual distributed coordination issue
- Ensure all processes properly synchronize
- Debug MPI communication problems directly
- Make distributed mode work reliably, not optionally

**Rationale**: Fallbacks hide real problems and create unreliable code that works sometimes but fails unpredictably. Fix the root cause instead of masking it.

## TASK SCALING FOR REAL DATA - ALWAYS USE 8 TASKS
CRITICAL: Real cosmological simulation data contains billions of particles (2.74B+ particles). Processing with insufficient tasks causes extremely slow execution.

**PARTICLE COUNTS**:
- lcdm-validation: 2.74 billion particles
- File sizes: 92GB+ 
- Memory per process: 15GB+ with full dataset

**MANDATORY TASK SCALING**:
- NEVER use 2 tasks for real data (takes hours to load and grid)
- ALWAYS use 8 tasks minimum for real data processing
- Use 2 tasks ONLY for synthetic/debug data with <1M particles

**CORRECT USAGE**:
```bash
# CORRECT: Real data with 8 tasks for proper scaling
./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256 --ntasks 8 --nodes 2

# WRONG: Real data with 2 tasks (will take hours)
./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256 --ntasks 2

# OK: Debug/synthetic data with 2 tasks
./run_examples.sh --debug-synthetic --assignment ngp --ngrid 64 --ntasks 2
```

**NODE SCALING**:
- 4 tasks: 1 node
- 8 tasks: 2 nodes (MANDATORY - each node supports max 4 tasks)
- 12+ tasks: Scale nodes accordingly (nodes = ceiling(tasks/4))

## REAL DATA TESTING PROTOCOL
CRITICAL: Real data tests must follow strict protocol for reproducibility and comparison.

**PRE-TESTING REQUIREMENTS**:
- MANDATORY: Clean commit all changes before real data testing
- Use `/claude/commands/commit-changes.md` command to commit current work
- Ensure working directory is clean before launching real data jobs

**DENSITY GRID GENERATION**:
- MANDATORY: Always use `--save-density-grid` flag for real data tests
- Required for method comparison and debugging
- Enables post-hoc analysis and validation

**TIME REQUIREMENTS FOR 256³ REAL DATA**:
- INSUFFICIENT: 30 minutes (times out during FFT phase)
- MINIMUM: 60 minutes for complete power spectrum calculation
- Real data (2.74B particles, 256³ grid): ~35s I/O, ~15s gridding, ~10-15min FFT+binning

**CORRECT REAL DATA EXECUTION**:
```bash
# 1. First commit all changes
/.claude/commands/commit-changes.md

# 2. Then run real data test with density grid saving - USE 60 MINUTES
./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256 --ntasks 8 --nodes 2 --save-density-grid --time 60
```

**RATIONALE**:
- Clean commits ensure reproducible results tied to specific code versions
- Density grids enable detailed comparison between NGP and CIC methods
- 60+ minute time limit ensures complete analysis including FFT phase
- Essential for N=5 outlier investigation and method validation