# Test Utilities

This directory contains utility scripts for managing test data and debugging.

## Files

- **`generate_test_data.py`** - Creates the strategic test data files used by the I/O tests
  - Only run this if test data needs to be regenerated
  - Generated files are committed to git in `tests/test_data/`
  
## Usage

```bash
# Regenerate all test data (rarely needed)
cd tests/
python utils/generate_test_data.py
```

## Test Data Structure

The generated test data includes:

- **Lightcone files** (`*.lcp.*`): 4 files with 16 particles each, covering different angular positions
- **Snapshot file** (`test_sim.00001`): 1 unified file with 45 particles across different spatial regions  
- **Halo files** (`*.fofstats.*`): 3 files with 6 halos each, varying masses from 10-5000 particles
- **Parameter file** (`test_sim.par`): Points to the output directory structure
- **Log file** (`test_sim.log`): Contains redshift output information

Total size: ~6KB (suitable for git repository)