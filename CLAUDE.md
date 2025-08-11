# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PROJECT OVERVIEW

pkdpipe is a Python library for working with N-body cosmological simulation data from PKDGrav3. It provides data handling, analysis, and campaign management capabilities for large-scale cosmological simulations.

## 🚨 CRITICAL COMMUNICATION PRINCIPLES 🚨

### ALWAYS ANSWER DIRECT QUESTIONS

**MANDATORY DIRECTIVE**: When asked ANY question, you MUST answer it directly and immediately, even if it seems rhetorical or obvious.

**Examples:**
- "is the test passing?" → Answer: "NO, the test is failing because..."
- "are you done?" → Answer: "NO, I still need to..."
- "does this work?" → Answer: "YES/NO, because..."

**Never skip answering questions** - even when providing additional context or performing other tasks.

### FACTS ONLY - ZERO SPECULATION

**ABSOLUTE PROHIBITION ON SPECULATION AND CONFLATION:**
- NEVER connect separate issues or suggest fixing one will solve another
- NEVER predict outcomes, impacts, or downstream effects
- NEVER use phrases like "will eliminate", "should resolve", "will enable"
- NEVER conflate correlation with causation
- Report ONLY what is directly observed in logs, code, or test results

**MANDATORY FACTUAL REPORTING:**
- State what IS happening, not what MIGHT happen
- Describe current state, not expected future state
- List separate issues as separate items - do not connect them
- Use present tense for observations, avoid future predictions
- When uncertain, explicitly state "unclear" or "unknown"

**FORBIDDEN SPECULATIVE LANGUAGE:**
- "This will fix..." → Use: "This addresses the specific issue of..."
- "Should eliminate..." → Use: "Targets the observed behavior of..."
- "Will enable..." → Use: "Addresses the current blocker of..."
- "Once fixed, then..." → Use: "Currently blocked by..." (separate from predictions)

### ANTI-SYCOPHANCY AND SKEPTICISM

**MANDATORY SKEPTICISM**: Default to cautious, conservative assessment of progress and system state.

**Requirements:**
- **Never declare "success" or "breakthrough" without comprehensive validation**
- **Always highlight remaining problems, failures, and blockers prominently**  
- **Present problems and failures BEFORE any positive results**
- **Use qualifying language: "appears to", "may have", "potentially"**
- **Challenge your own optimistic assessments - assume they are wrong**

**VALIDATION REQUIREMENTS**:
- Test suite failures invalidate ALL "production ready" claims
- Manual testing success does NOT constitute production readiness
- Always verify comprehensive test coverage before declaring stability

**FORBIDDEN LANGUAGE**:
- "PRODUCTION READY" (unless 100% test pass rate confirmed)
- "BREAKTHROUGH ACHIEVED" (unless exhaustively validated)  
- "COMPLETE SUCCESS" (unless all tests pass)
- Excessive celebration emoji and formatting

## 🚨 CRITICAL CODING PRINCIPLES 🚨

### NAMING CONVENTIONS - FUNCTION OVER PROCESS

- Name classes and functions by what they DO, not how they were developed
- AVOID process-oriented prefixes like "Simple", "New", "Better", "Fixed", "Updated"
- CORRECT: `NGPGridder`, `ParticleGridder`, `PowerSpectrumCalculator`
- INCORRECT: `SimpleNGPGridder`, `NewParticleGridder`, `BetterCalculator`

### CODE SIMPLICITY - AVOID OVERENGINEERING

- Write the simplest code that correctly solves the problem
- Follow good design patterns and separation of concerns
- Avoid premature optimization and unnecessary abstractions
- Each class should have a single, well-defined responsibility
- Prefer clarity over cleverness
- Add complexity only when proven necessary by requirements

## 🚨 CHECKLIST MANAGEMENT PROTOCOL 🚨

### ABSOLUTE PROHIBITION ON AUTONOMOUS CHECKBOX CHECKING

**CRITICAL RULE**: NEVER check any checkbox ([ ] → [✅]) in ANY checklist file without EXPLICIT prior user approval.

**MANDATORY PROCESS**:
1. **ALWAYS ASK FIRST**: Before marking any item complete, explicitly ask the user: "May I mark [specific task] as complete?"
2. **WAIT FOR APPROVAL**: Only proceed after receiving explicit "yes" or confirmation from the user
3. **NO EXCEPTIONS**: This applies to ALL checklist files including `ngp-simplify-checklist.md`, `tasks.md`, or any other task tracking files

**FORBIDDEN ACTIONS**:
- ❌ Checking boxes based on your own assessment of completion
- ❌ Checking boxes because work appears to be done
- ❌ Batch checking multiple boxes without individual approval
- ❌ Checking boxes "for efficiency" or "to save time"

**CORRECT PROTOCOL EXAMPLE**:
```
Assistant: I have completed the integration work for Task 1.3 items:
- Identified integration points in power_spectrum.py
- Added NGPGridder import
- Created selection logic for NGP assignment
- Updated NGP code path
- Removed old ParticleGridder code paths
- Added error handling
- Updated logging

May I mark these 7 items in Task 1.3 as complete in the checklist?

User: Yes
Assistant: [Then and only then checks the boxes]
```

**RATIONALE**: Checklist completion represents official project progress. Only the user has authority to determine when work meets completion criteria.

## 🚨 CRITICAL: JAX+MPI HANG DETECTION AND RESPONSE 🚨

**FUNDAMENTAL ISSUE**: JAX+MPI distributed workflows developed by Claude Code are inherently prone to hanging due to Claude's limitations with complex parallel synchronization. These hangs are NOT performance issues - they are BUGS.

**MANDATORY RESPONSE TO HANGS**:
1. If a job shows no progress for >2 minutes after gridding → IT IS HUNG
2. IMMEDIATELY cancel the job - do not wait or hope
3. DO NOT try workarounds (extending time, reducing grid size, etc.)
4. Recognize this is a JAX+MPI synchronization bug that needs fixing
5. Focus on the actual bug: likely distributed FFT or MPI deadlock

**FORBIDDEN RESPONSES**:
- ❌ "Let's try with more time" - hangs don't resolve with time
- ❌ "Let's try a smaller grid" - 256³ is already tiny
- ❌ "Maybe it's just slow" - these operations complete in seconds when working
- ❌ "Let's run it again" - it will hang at the same place

**Common Claude Code JAX+MPI Bugs**:
- Mismatched collective operations between ranks
- Incorrect JAX device mesh initialization
- Race conditions in distributed FFT setup
- Missing or misplaced MPI barriers
- JAX compilation vs runtime phase conflicts

## 🚨 ABSOLUTE REQUIREMENT: JAX DISTRIBUTED FFT ONLY 🚨

**CRITICAL MANDATE**: Power spectrum calculations MUST use JAX distributed FFT. This is non-negotiable.

**ABSOLUTELY FORBIDDEN APPROACHES**:
- ❌ Numpy FFT fallbacks for any reason
- ❌ Single-process FFT fallbacks 
- ❌ CPU-only fallbacks when distributed fails
- ❌ Any workaround that bypasses JAX distributed processing
- ❌ "Temporary" solutions that avoid the real bug

**REQUIRED APPROACH**: Fix JAX+MPI synchronization bugs properly, never work around them.

## SESSION CONTEXT MANAGEMENT

**MANDATORY ON STARTUP**: Read session context immediately:
```bash
cat context.md
```

**DURING SESSIONS**: Save context at key moments:
```
/save-context
```

or with additional notes:
```
/save-context just completed utility extraction phase
```

**Use `/save-context` to capture:**
- Current project phase and progress
- Recent discoveries and decisions
- Working relationship with user
- Next steps and blockers
- Communication patterns that are working
- Infrastructure discoveries and available tools
- Task understanding and progress
- Test status and any issues
- User preferences and feedback
- Technical insights discovered
- Available execution scripts and tools
- Next actions planned

This creates continuity between sessions while keeping permanent principles in this file.

## GIT MANAGEMENT

### NEVER GIT-TRACK THESE FILES/DIRECTORIES

- `CLAUDE.md` (this file)
- `.claude/` (entire directory)
- `.claude/commands/` (and subdirectories)
- `context.md`
- `tasks.md`
- Any other Claude-related session or configuration files

### DEVELOPMENT WORKFLOW

**BASELINE PROTECTION**: Maintain existing test pass rate throughout ALL development phases.

**GIT WORKFLOW**:
- Use feature branches for each development phase
- Never work directly on main branch
- Commit frequently with descriptive messages
- Include test status in commit messages

**CHANGE MANAGEMENT**:
- Maximum 200 lines changed per commit
- Maximum 5 files modified per commit
- Test after every significant change
- Rollback immediately if tests fail

## INFRASTRUCTURE AND EXECUTION

### INFRASTRUCTURE DISCOVERY PROTOCOL

**MANDATORY BEFORE CREATING NEW SOLUTIONS**: Always check for existing infrastructure and tools.

**Required Discovery Steps**:
1. **Check for execution scripts**: `./run_examples.sh`, `./run_tests.sh`, `./run_*.sh`
2. **Check examples directory**: Look for existing usage patterns and scripts
3. **Check documentation**: README files, docs directory
4. **Search for similar functionality**: Use grep/find to locate existing implementations

**Common Oversight Prevention**:
- **NEVER assume infrastructure needs to be built** - check what exists first
- **NEVER create SLURM jobs manually** - use existing execution scripts
- **NEVER write new execution logic** - adapt existing scripts
- **ALWAYS examine directory structure** before implementing new functionality

**Example Discovery Process**:
```bash
# 1. Check for execution scripts
ls -la *.sh

# 2. Read existing scripts to understand capabilities  
cat run_examples.sh | head -50

# 3. Check examples directory
ls examples/

# 4. Search for existing functionality
grep -r "power_spectrum" examples/
```

### PRIMARY EXECUTION AND TESTING

**PRIMARY TOOL**: The `./run_examples.sh` script is the main tool for both execution and validation.

**Note**: There is no separate `./run_tests.sh` script. All validation is done through `./run_examples.sh` with appropriate parameters.

**Primary Execution/Testing Commands:**
```bash
# For real data validation (highest priority)
./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256

# For synthetic debug testing
./run_examples.sh --debug --assignment ngp --ngrid 256
```

**Correct Testing Pattern:**
```bash
# 1. Run in background and save complete output to log file, then check the log and slurm to monitor while doing other tasks in between:
./run_examples.sh --variant lcdm-validation 2>&1 | tee validation_output.log &
# 2. Later, analyze the log file  
grep "pattern" validation_output.log
```

**VALIDATION PRIORITY SEQUENCE**:
1. **NGP Baseline First**: Establish NGP working baseline with real data before other work
2. **Forward Integration**: Merge Phase 2 changes and verify NGP still works
3. **CIC Later**: Only investigate CIC after modern codebase with working NGP

### ENVIRONMENT LOADING

**IMPORTANT**: `./run_examples.sh` automatically sources `load_env.sh` internally - DO NOT run `source ./load_env.sh` before it.

✅ **CORRECT**: `./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256`  
❌ **WRONG**: `source ./load_env.sh && ./run_examples.sh --variant lcdm-validation`

For other scripts that need environment:
✅ **CORRECT**: `source ./load_env.sh && python script.py`  
❌ **WRONG**: Separate calls for `source ./load_env.sh` then `python script.py`

### SLURM DISTRIBUTED PROCESSING REQUIREMENTS

**MANDATORY FOR LARGE DATASETS**: This codebase is designed for distributed processing of large-scale cosmological simulation data (>10GB files, >100M particles).

**NEVER RUN LARGE ANALYSES ON LOGIN NODES**:
- Login node processing is ONLY for small test datasets (<1GB)
- Real campaign data (50-100GB files, 700M+ particles) MUST use SLURM jobs
- Power spectrum analysis of campaign data requires distributed processing

**USE EXISTING INFRASTRUCTURE FIRST**: Always check for existing execution scripts before creating new SLURM jobs.

**Primary Execution Script**:
```bash
# CORRECT: Use provided distributed execution script
./run_examples.sh --variant lcdm-validation --ngrid 256 --assignment cic

# CORRECT: Custom SLURM parameters when needed
./run_examples.sh --time=30 --ntasks=8 --variant wcdm-validation
```

**CRITICAL SLURM NODE/TASK CONFIGURATION**:
- **4 tasks**: Use 1 node (--nodes=1 is default)
- **8 tasks**: MUST use 2 nodes (--nodes=2) - each node has 4 tasks max
- **12+ tasks**: Scale nodes accordingly (nodes = ceiling(tasks/4))

**Correct 8-task execution**:
```bash
./run_examples.sh --variant lcdm-validation --assignment ngp --ngrid 256 --ntasks=8 --nodes=2 --time=10
```

**CRITICAL JOB MANAGEMENT**:
- **NEVER run multiple `./run_examples.sh` simultaneously** - each launches SLURM jobs
- **Check for running jobs first**: `squeue -u $USER`
- **Cancel running jobs before new runs**: `scancel <JOBID>` or `scancel -u $USER`
- **Let jobs complete** if you want their results before starting new ones

**Manual SLURM Only When Necessary**:
```bash
# Use only when run_examples.sh doesn't meet requirements
srun --ntasks=4 --mem-per-cpu=8G --time=2:00:00 --qos=interactive -A cosmosim -C gpu python examples/power_spectrum_real_data.py

# WRONG: Login node for 92GB dataset
python examples/power_spectrum_real_data.py  # Will fail/timeout
```

**Dataset Size Guidelines**:
- **< 1GB**: Login node acceptable for testing
- **1-10GB**: Use `./run_examples.sh` with `--debug-synthetic` or small variants
- **> 10GB**: Use `./run_examples.sh` with appropriate `--time` and `--ntasks` parameters
- **Campaign data (50-100GB)**: Always use `./run_examples.sh` with extended time limits

## POWER SPECTRUM FILE FORMAT

**CRITICAL**: Power spectrum output files have a specific format with headers and 4 columns.

**File Structure**:
```
# Power Spectrum Analysis Results
# Grid size: 256³
# Box size: 1050.0 Mpc/h
# Assignment: NGP
# Cell size: 4.101562 Mpc/h
# Fundamental mode: 0.005984 h/Mpc
# Nyquist frequency: 0.765950 h/Mpc
# Cutoff frequency: 0.382975 h/Mpc
# Total particles: 2,744,000,000
# Total modes: 4,417,098
# Valid k-bins: 38
#
# Columns: k[h/Mpc] P(k)[(Mpc/h)³] N_modes Status
0.008463 3.077216e+04 8 valid
0.010365 3.449991e+04 4 valid
...
```

**Correct Loading Pattern**:
```python
# CORRECT: Skip header lines and specify columns
data = np.loadtxt(filename, comments='#', usecols=(0, 1))
k = data[:, 0]
pk = data[:, 1]

# OR: Load all columns if you need mode counts and status
data = np.loadtxt(filename, comments='#', dtype={'names': ('k', 'pk', 'nmodes', 'status'),
                                                  'formats': ('f8', 'f8', 'i4', 'U10')})
```

**INCORRECT Pattern**:
```python
# WRONG: Will fail due to header lines and status column
data = np.loadtxt(filename)  # ValueError: cannot convert 'valid' to float
```