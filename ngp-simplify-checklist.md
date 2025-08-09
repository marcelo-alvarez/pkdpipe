# NGP Simplification Checklist

**Reference Document**: `ngp-simplify.md` - Complete strategy and implementation plan  
**Purpose**: Track progress through NGP simplification implementation  
**Status**: 🎯 READY TO BEGIN - NGP baseline established

**⚠️ CRITICAL: CHECKLIST COMPLETION PROTOCOL**
Items in this checklist may only be checked off as complete with PRIOR AND IMMEDIATE user approval. Never mark checklist items as done without explicit user confirmation. Always request permission before marking any item as ✅ COMPLETE.

## Phase 1: Simple NGP Implementation ⏳ IN PROGRESS

### Task 1.1: Create NGPGridder Class
- [✅] **Create file**: `pkdpipe/ngp_gridder.py`
- [✅] **Implement class structure**: Basic `NGPGridder` class with `__init__`
- [✅] **Add slab decomposition logic**: `z_start`, `z_end` calculation based on rank
- [✅] **Implement particle binning**: Direct floor division for grid coordinates
- [✅] **Add particle filtering**: Keep only particles in process's z-slabs
- [✅] **Implement local grid creation**: Allocate local density grid array
- [✅] **Add particle-to-grid assignment**: Simple binning loop
- [✅] **Add particle counting**: Track particles processed per process
- [✅] **Implement MPI reduction**: Combine local grids to full density grid
- [✅] **Add validation methods**: Particle count verification functions

### Task 1.2: Unit Tests for NGPGridder  
- [✅] **Create test file**: `tests/test_ngp_gridder.py`
- [✅] **Test class initialization**: Verify slab bounds calculation
- [✅] **Test single particle**: Verify correct grid cell assignment
- [✅] **Test particle filtering**: Verify process-local particle selection
- [✅] **Test particle conservation**: Verify no particles lost/duplicated
- [✅] **Test grid bounds**: Verify all particles assigned within grid
- [✅] **Test synthetic data**: Small known particle distributions
- [✅] **Test edge cases**: Verify particles exactly on grid boundaries
- [✅] **Test MPI functionality**: Multi-process grid reduction

### Task 1.3: Integration with PowerSpectrumCalculator
- [ ] **Identify integration points**: Where NGP gridding is called in `power_spectrum.py`
- [ ] **Add NGPGridder import**: Import new class
- [ ] **Create selection logic**: Use NGPGridder when `assignment='ngp'`
- [ ] **Update NGP code path**: Replace old gridding with NGPGridder
- [ ] **Remove old code paths**: Delete ParticleGridder after validation complete
- [ ] **Add error handling**: Proper error messages for unsupported cases
- [ ] **Update logging**: Add NGPGridder-specific log messages

## Phase 2: Validation Against Current Results ⏳ PENDING

### Task 2.1: Use Existing Baseline Reference Files
- [✅] **Baseline data available**: Located in `./ngp-complex-validation-data/`
- [✅] **4-task power spectrum**: `power_spectrum_ngrid256_ngp_ntasks4_lcdm_validation_97dbfad3_20250809_132817.txt`
- [✅] **8-task power spectrum**: `power_spectrum_ngrid256_ngp_ntasks8_lcdm_validation_97dbfad3_20250809_132659.txt`  
- [✅] **4-task density**: `density_grid_ngrid256_ntasks4_97dbfad-dirty_20250809_132807.bin`
- [✅] **8-task density**: `density_grid_ngrid256_ntasks8_97dbfad-dirty_20250809_132648.bin`
- [✅] **Parameters documented**: lcdm-validation, ngp, ngrid=256, exact particle conservation verified
- [✅] **Baseline quality verified**: Perfect particle conservation (2,744,000,000) confirmed

### Task 2.2: New Implementation Testing - **REAL DATA ONLY**
- [ ] **Run 4-task test**: `./run_examples.sh --ngrid 256 --assignment ngp --variant lcdm-validation --ntasks=4 --save-density-grid`
- [ ] **Run 8-task test**: `./run_examples.sh --ngrid 256 --assignment ngp --variant lcdm-validation --ntasks=8 --nodes=2 --save-density-grid`
- [ ] **Compare particle counts**: Must be exactly 2,744,000,000 for both runs
- [ ] **Compare density grid sums**: Verify total particle preservation
- [ ] **Compare power spectra**: Against baseline files in `./ngp-complex-validation-data/`
- [ ] **Verify 4-task vs 8-task consistency**: Results should be identical
- [ ] **Check numerical precision**: Document any floating-point differences

### Task 2.3: Success Criteria Verification
- [ ] **Exact particle conservation**: 2,744,000,000 particles in all runs
- [ ] **Density grid sum match**: Within numerical precision of baseline
- [ ] **Power spectra match**: Within statistical significance thresholds
- [ ] **4-task/8-task consistency**: New implementation produces identical results
- [ ] **Performance acceptable**: Runtime within reasonable bounds of current implementation
- [ ] **No crashes or errors**: All test cases complete successfully

## Phase 3: Production Integration ⏳ PENDING

### Task 3.1: Replace Production NGP Code Paths
- [ ] **Update default behavior**: NGPGridder becomes primary NGP implementation
- [ ] **Delete old code**: Remove ParticleGridder and old infrastructure from git and disk
- [ ] **Update documentation**: Reflect new implementation in code comments
- [ ] **Update example scripts**: Ensure all examples work with new implementation
- [ ] **Test with all variants**: lcdm-validation, wcdm-validation, phicdm-validation
- [ ] **Verify SLURM integration**: All `./run_examples.sh` parameters work correctly

### Task 3.2: Code Cleanup and Documentation
- [ ] **Clean up imports**: Remove unused old gridding imports where applicable
- [ ] **Update class docstrings**: Document NGPGridder thoroughly
- [ ] **Add inline comments**: Explain key implementation decisions
- [ ] **Update CLAUDE.md**: Reflect new NGP implementation status
- [ ] **Update context.md**: Mark NGP simplification complete
- [ ] **Create summary**: Document what was accomplished and lessons learned

### Task 3.3: Final Validation
- [ ] **Full regression test**: Run all existing test commands with new implementation
- [ ] **Compare to original baseline**: Verify final implementation matches original baseline
- [ ] **Performance benchmark**: Document any performance changes
- [ ] **Memory usage check**: Verify memory usage is acceptable
- [ ] **Clean git status**: Ensure no temporary files left in repository

## Future Phase: CIC Evaluation (DEFERRED) 🔴 BLOCKED

### Task 4.1: CIC Necessity Assessment (After NGP Complete)
- [ ] **Scientific requirements review**: Is CIC actually needed for campaign validation?
- [ ] **Accuracy analysis**: What improvement does CIC provide over NGP?
- [ ] **Complexity cost assessment**: Is CIC worth the additional complexity?
- [ ] **Decision point**: Proceed with CIC implementation or stay with NGP

### Task 4.2: CICGridder Implementation (If Needed)
- [ ] **Design CICGridder**: Clean CIC implementation separate from NGP
- [ ] **Implement ghost cell handling**: Explicit, well-documented ghost cell logic for CIC interpolation
- [ ] **Add CIC-specific tests**: Test suite for CIC functionality
- [ ] **Integration testing**: CIC and NGP working independently
- [ ] **Validation against theory**: CIC results match expected theoretical behavior

## Status Legend
- 🎯 **READY TO BEGIN**: Prerequisites met, can start immediately
- ⏳ **IN PROGRESS**: Currently being worked on
- ⏳ **PENDING**: Waiting for prerequisite completion
- ✅ **COMPLETE**: Task finished and verified
- 🔴 **BLOCKED**: Cannot proceed due to dependency or issue
- 🔴 **DEFERRED**: Deliberately postponed for later

## Notes
- **Baseline Status**: NGP working perfectly (commit 97dbfad), exact particle conservation achieved
- **Current Implementation**: Complex but functional, serves as validation baseline
- **Strategy**: Simplify NGP first, evaluate CIC necessity later
- **Success Criteria**: Identical results to current baseline, simplified maintainable code