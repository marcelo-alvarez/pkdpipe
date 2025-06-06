# Cosmosim Mocks 2025 Campaign

**Campaign ID:** `cosmosim-mocks-2025`  
**Status:** **DEADLINE: July 1, 2025**  
**Configuration:** [`cosmosim-mocks-2025.yaml`](./cosmosim-mocks-2025.yaml)

## Overview

Multi-cosmology simulation campaign with LCDM, wCDM, and phiCDM models for summer cosmological mock generation. This campaign supports upcoming surveys and cosmological parameter inference studies with a **critical deadline of July 1, 2025** for production runs.

## Campaign Settings

- **Max Concurrent Jobs:** 6 (Conservative limit for Perlmutter GPU partition)
- **Global Deadline:** July 1, 2025
- **Target System:** NERSC Perlmutter (GPU partition)

## Storage Locations

- **Output Directory:** `/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025`
- **Scratch Directory:** `/pscratch/sd/m/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025`

## Cosmology Models

This campaign uses three clean cosmology presets:

| Model | Preset | Description |
|-------|--------|-------------|
| **ΛCDM** | `lcdm` | Standard Lambda Cold Dark Matter cosmology |
| **wCDM** | `wcdm` | Dark energy with evolving equation of state (w0, wa) |
| **φCDM** | `phicdm` | Scalar field dark energy model |

All cosmology presets inherit from `desi-dr2-planck-act-mnufree` baseline parameters.

## Resolution Configurations

| Preset | Box Size | Particles | Purpose | Scale Factor |
|--------|----------|-----------|---------|--------------|
| `S0-validation` | 1050 Mpc/h | 1400³ | Quick validation runs | 1x |
| `S0-scaling` | 2100 Mpc/h | 2800³ | Scaling performance tests | 2x |
| `S0-highres` | 3150 Mpc/h | 4200³ | High-resolution tests | 3x |
| `S0-production` | 5250 Mpc/h | 7000³ | Full production simulations | 5x |

## Simulation Variants

### Phase 1: Validation Runs (Priority 100-90)
**Timeline:** June 10-14, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-validation` | ΛCDM | S0-validation | 100 | Jun 10 | None |
| `wcdm-validation` | wCDM | S0-validation | 95 | Jun 12 | lcdm-validation |
| `phicdm-validation` | φCDM | S0-validation | 90 | Jun 14 | wcdm-validation |

**Purpose:** Parameter and workflow testing with full 39 snapshots (ΛCDM) or 3 snapshots (wCDM, φCDM).

### Phase 2: Scaling Tests (Priority 89-80)
**Timeline:** June 20, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-scaling` | ΛCDM | S0-scaling | 85 | Jun 20 | lcdm-validation |

**Purpose:** Performance scaling validation with full 39 snapshots at intermediate resolution.

### Phase 3: Production Runs (Priority 79-70) **CRITICAL**
**Timeline:** June 28 - July 1, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-production` | ΛCDM | S0-production | 79 | Jun 28 | lcdm-scaling |
| `wcdm-production` | wCDM | S0-production | 75 | Jun 30 | lcdm-production |

**Purpose:** **CRITICAL** full production runs with 39 snapshots that **MUST** be submitted by July 1, 2025 deadline.

### Phase 4: Extended Analysis (Priority 69-60)
**Timeline:** July 5, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `phicdm-production` | φCDM | S0-production | 65 | Jul 5 | wcdm-production |

**Purpose:** Lower priority production run with 39 snapshots, submit only after critical runs are secured.

## Dependency Chain

```
lcdm-validation
├── wcdm-validation
│   └── phicdm-validation
└── lcdm-scaling
    └── lcdm-production
        └── wcdm-production (CRITICAL)
            └── phicdm-production
```

## Output Snapshots

### Full 39-Snapshot Runs
- **Validation:** `lcdm-validation`
- **Scaling:** `lcdm-scaling`
- **Production:** `lcdm-production`, `wcdm-production`, `phicdm-production`
- **Extended:** `lcdm-highres`

**Redshift Range:** z=3.0 → z=0.0 with 39 carefully selected redshifts for comprehensive analysis.

### Quick 3-Snapshot Runs
- **Validation:** `wcdm-validation`, `phicdm-validation`

**Redshifts:** z=3.0, z=1.0, z=0.0

## Critical Timeline

| Date | Milestone | Action Required |
|------|-----------|----------------|
| **June 10** | Validation begins | Submit `lcdm-validation` |
| **June 14** | Validation complete | All validation runs submitted |
| **June 20** | Scaling complete | `lcdm-scaling` submitted |
| **June 28** | Production begins | Submit `lcdm-production` |
| **June 30** | wCDM production | Submit `wcdm-production` |
| **July 1** | **DEADLINE** | **Both production runs MUST be submitted** |

## Usage

### Create Campaign
```bash
pkdpipe-campaign create campaigns/cosmosim-mocks-2025.yaml
```

### Monitor Progress
```bash
pkdpipe-campaign status campaigns/cosmosim-mocks-2025.yaml
```

### List All Variants
```bash
pkdpipe-campaign list campaigns/cosmosim-mocks-2025.yaml
```

### Submit Next Ready Jobs
```bash
pkdpipe-campaign submit campaigns/cosmosim-mocks-2025.yaml
```

## Risk Assessment

- **HIGH RISK:** July 1 deadline is extremely tight
- **MITIGATION:** Prioritized submission order ensures critical runs go first
- **CONTINGENCY:** φCDM production can be delayed if resources are constrained
- **MONITORING:** Daily status checks recommended after June 25

## Technical Specifications

- **LPT Order:** 3rd order Lagrangian Perturbation Theory
- **Initial Redshift:** z=12.0
- **Halo Finding:** Enabled (`bDoHalo: 1`)
- **Frame Dumps:** Enabled (`bDumpFrame: 1`)
- **Gas/Stars:** Disabled (dark matter only)

---

**Generated:** June 3, 2025  
**Campaign Configuration:** [`cosmosim-mocks-2025.yaml`](./cosmosim-mocks-2025.yaml)  
**Documentation:** [`campaign-management.md`](../docs/campaign-management.md)
