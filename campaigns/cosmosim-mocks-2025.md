# Cosmosim Mocks 2025 Campaign

**Campaign ID:** `cosmosim-mocks-2025`  
**Status:** ⚠️ **CRITICAL DEADLINE: July 1, 2025**  
**Configuration:** [`cosmosim-mocks-2025.yaml`](./cosmosim-mocks-2025.yaml)

## Overview

Multi-cosmology simulation campaign with LCDM, wCDM, and phiCDM models for summer cosmological mock generation. This campaign supports upcoming surveys and cosmological parameter inference studies with a **critical deadline of July 1, 2025** for production runs.

## Campaign Settings

- **Max Concurrent Jobs:** 6 (Conservative limit for Perlmutter GPU partition)
- **Global Deadline:** July 1, 2025 23:59:59 UTC
- **Target System:** NERSC Perlmutter (GPU partition)
- **Account:** `cosmosim`
- **Contact:** marcelo.alvarez@stanford.edu

## Storage Locations

- **Output Directory:** `/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025`
- **Scratch Directory:** `/pscratch/sd/m/malvarez/pkdgrav3/campaigns/cosmosim-mocks-2025`

## Cosmology Models

This campaign uses three summer cosmology presets:

| Model | Preset | Description |
|-------|--------|-------------|
| **ΛCDM** | `summer-lcdm` | Standard Lambda Cold Dark Matter cosmology |
| **wCDM** | `summer-wcdm` | Dark energy with constant equation of state parameter w |
| **φCDM** | `summer-phicdm` | Scalar field dark energy model |

All cosmology presets inherit from `desi-dr2-planck-act-mnufree` baseline parameters.

## Resolution Configurations

| Preset | Box Size | Particles | Purpose |
|--------|----------|-----------|---------|
| `summer-validation` | 1050 Mpc/h | 1400³ | Quick validation runs |
| `summer-scaling-2800` | 2100 Mpc/h | 2800³ | Scaling performance tests |
| `summer-scaling-4200` | 3150 Mpc/h | 4200³ | High-resolution scaling tests |
| `summer-production` | 5250 Mpc/h | 7000³ | Full production simulations |

## Simulation Variants

### Phase 1: Validation Runs (Priority 100-90)
**Timeline:** June 10-14, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-validation` | ΛCDM | Validation | 100 | Jun 10 | None |
| `wcdm-validation` | wCDM | Validation | 95 | Jun 12 | lcdm-validation |
| `phicdm-validation` | φCDM | Validation | 90 | Jun 14 | wcdm-validation |

**Purpose:** Parameter and workflow testing with full 39 snapshots (ΛCDM) or 3 snapshots (wCDM, φCDM).

### Phase 2: Scaling Tests (Priority 89-80)
**Timeline:** June 20-25, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-scaling-2800` | ΛCDM | 2800³ | 85 | Jun 20 | lcdm-validation |
| `lcdm-scaling-4200` | ΛCDM | 4200³ | 80 | Jun 25 | lcdm-scaling-2800 |

**Purpose:** Performance scaling validation with full 39 snapshots at intermediate resolutions.

### Phase 3: Production Runs (Priority 79-70) ⚠️ **CRITICAL**
**Timeline:** June 28 - July 1, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `lcdm-production` | ΛCDM | Production | 79 | Jun 28 | lcdm-scaling-4200 |
| `wcdm-production` | wCDM | Production | 75 | Jun 30 | lcdm-production |

**Purpose:** **CRITICAL** full production runs that **MUST** be submitted by July 1, 2025 deadline.

### Phase 4: Extended Analysis (Priority 69-60)
**Timeline:** July 5, 2025

| Variant | Cosmology | Resolution | Priority | Deadline | Dependencies |
|---------|-----------|------------|----------|----------|--------------|
| `phicdm-production` | φCDM | Production | 65 | Jul 5 | wcdm-production |

**Purpose:** Lower priority production run, submit only after critical runs are secured.

## Dependency Chain

```
lcdm-validation
├── wcdm-validation
│   └── phicdm-validation
└── lcdm-scaling-2800
    └── lcdm-scaling-4200
        └── lcdm-production
            └── wcdm-production ⚠️ (CRITICAL)
                └── phicdm-production
```

## Output Snapshots

### Full 39-Snapshot Runs
- **Validation:** `lcdm-validation`
- **Scaling:** `lcdm-scaling-2800`, `lcdm-scaling-4200`
- **Production:** `lcdm-production`, `wcdm-production`, `phicdm-production`

**Redshift Range:** z=3.0 → z=0.0 with additional specific redshifts for analysis.

### Quick 3-Snapshot Runs
- **Validation:** `wcdm-validation`, `phicdm-validation`

**Redshifts:** z=3.0, z=1.0, z=0.0

## Critical Timeline

| Date | Milestone | Action Required |
|------|-----------|----------------|
| **June 10** | Validation begins | Submit `lcdm-validation` |
| **June 14** | Validation complete | All validation runs submitted |
| **June 25** | Scaling complete | `lcdm-scaling-4200` submitted |
| **June 28** | Production begins | Submit `lcdm-production` ⚠️ |
| **June 30** | wCDM production | Submit `wcdm-production` ⚠️ |
| **July 1** | **DEADLINE** | **Both production runs MUST be submitted** |

## Usage

### Create Campaign
```bash
pkdpipe-campaign create campaigns/cosmosim-mocks-2025.yaml
```

### Monitor Progress
```bash
pkdpipe-campaign status cosmosim-mocks-2025
```

### List All Variants
```bash
pkdpipe-campaign list cosmosim-mocks-2025
```

### Submit Next Ready Jobs
```bash
pkdpipe-campaign submit cosmosim-mocks-2025
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
