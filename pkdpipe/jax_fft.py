from typing import Callable, Any
import gc
import os
import numpy as np
from typing import Callable, Any

def _numpy_fft_fallback(x_np, direction='r2c'):
    """
    NumPy FFT fallback when JAX initialization fails.
    
    This provides a basic FFT implementation using NumPy when JAX is unavailable.
    Note: This doesn't support distributed FFT, only single-process mode.
    """
    print("Using NumPy FFT fallback (no distribution support)", flush=True)
    
    if direction == 'r2c':
        # Real-to-complex FFT
        return np.fft.rfftn(x_np)
    elif direction == 'c2r':
        # Complex-to-real inverse FFT
        return np.fft.irfftn(x_np)
    elif direction == 'c2c':
        # Complex-to-complex FFT
        return np.fft.fftn(x_np)
    else:
        raise ValueError(f"Unknown FFT direction: {direction}")

# JAX imports deferred until fft() is called to avoid multiprocessing conflicts
# This ensures NO JAX initialization happens during particle gridding/multiprocessing

def fft(x_np, direction='r2c'):
    """
    JAX-based distributed FFT with proper initialization order.
    
    CRITICAL ARCHITECTURE POINT:
    =========================== 
    This function is the SINGLE POINT where:
    1. JAX gets imported and initialized 
    2. JAX distributed mode gets set up
    3. NumPy arrays from CPU gridding get converted to JAX arrays
    
    The input `x_np` is a NumPy density grid that was computed entirely on CPU 
    using multiprocessing-safe operations. This function converts it to JAX
    arrays for GPU-based FFT operations.
    
    MEMORY TRANSFER POINT: x_np (CPU NumPy) → jax.device_put() → GPU JAX arrays
    
    Args:
        x_np: NumPy density grid from CPU-based particle gridding (IMPORTANT: CPU-only up to this point)
        direction: 'r2c' for real-to-complex FFT, 'c2r' for complex-to-real
        
    Returns:
        Local FFT result as NumPy array (converted back from JAX)
    """
    
    # CRITICAL: Check if distributed mode is needed FIRST, before any JAX operations
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    is_distributed = slurm_ntasks and int(slurm_ntasks) > 1
    
    if is_distributed:
        # Initialize JAX distributed mode BEFORE importing ANY other JAX modules
        print("Initializing JAX distributed mode in fft()...", flush=True)
        
        try:
            # CRITICAL: Only import jax.distributed, no other JAX modules yet
            import jax.distributed
            
            coordinator_address = os.environ.get('SLURM_STEP_NODELIST', 'localhost').split(',')[0]
            # Clean up SLURM nodelist format
            if '[' in coordinator_address:
                coordinator_address = coordinator_address.split('[')[0] + coordinator_address.split('[')[1].split('-')[0].replace(']', '')
            
            # Initialize distributed mode BEFORE any other JAX operations
            jax.distributed.initialize(
                coordinator_address=f"{coordinator_address}:63025",
                num_processes=int(slurm_ntasks),
                process_id=int(os.environ.get('SLURM_PROCID', 0))
            )
            print(f"JAX distributed initialized successfully", flush=True)
            
        except Exception as e:
            print(f"JAX distributed initialization failed: {e}", flush=True)
            raise RuntimeError("JAX distributed initialization failed - cannot proceed with distributed FFT") from e
    
    # NOW safe to import other JAX modules after distributed mode is set up
    try:
        import jax
        from jax import jit
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.experimental.custom_partitioning import custom_partitioning
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        
        print(f"JAX modules imported successfully", flush=True)
        
        # Set JAX configuration to match main branch behavior
        jax.config.update("jax_enable_x64", False)  # Use 32-bit precision
        # Do NOT set jax_platform_name - let JAX auto-detect from JAX_PLATFORMS env var
        
        # CRITICAL: Test JAX backend detection BEFORE using devices
        # Handle CUDA initialization failure gracefully
        try:
            backend = jax.default_backend()
            devices = jax.devices()
            print(f"JAX backend detected: {backend}", flush=True)
            print(f"JAX devices available: {devices}", flush=True)
        except RuntimeError as e:
            if "No visible GPU devices" in str(e) or "CUDA" in str(e):
                # CUDA failed, try to force CPU backend
                print(f"CUDA initialization failed: {e}", flush=True)
                print("Attempting to force CPU backend...", flush=True)
                try:
                    # Force CPU backend by overriding environment
                    os.environ['JAX_PLATFORMS'] = 'cpu'
                    print("Forcing JAX to reinitialize with CPU backend...", flush=True)
                    
                    # Clear JAX's cached backend state more thoroughly
                    import sys
                    if 'jax._src.xla_bridge' in sys.modules:
                        xla_bridge = sys.modules['jax._src.xla_bridge']
                        if hasattr(xla_bridge, '_backends'):
                            xla_bridge._backends.clear()
                            print("Cleared JAX backend cache", flush=True)
                        if hasattr(xla_bridge, '_backend_lock'):
                            # Reset the lock to allow reinitialization
                            pass
                    
                    # Force jax to re-detect platforms
                    if hasattr(jax, 'config'):
                        try:
                            jax.config.update('jax_platforms', 'cpu')
                            print("Updated JAX config to use CPU", flush=True)
                        except:
                            pass
                    
                    # Try getting backend with explicit platform
                    try:
                        backend = jax.lib.xla_bridge.get_backend('cpu')
                        devices = jax.devices('cpu')
                        print(f"JAX fallback to CPU backend: {backend.platform}", flush=True)
                        print(f"JAX CPU devices available: {devices}", flush=True)
                    except:
                        # Final fallback - try default again
                        backend = jax.default_backend()
                        devices = jax.devices()
                        print(f"JAX backend after CPU fallback: {backend}", flush=True)
                        print(f"JAX devices after fallback: {devices}", flush=True)
                        
                except Exception as fallback_error:
                    print(f"CPU fallback also failed: {fallback_error}", flush=True)
                    # Don't give up yet - try one more time with a clean import
                    try:
                        print("Last resort: attempting NumPy FFT fallback", flush=True)
                        return _numpy_fft_fallback(x_np, direction)
                    except:
                        raise RuntimeError(f"JAX initialization failed for both CUDA and CPU: {e}")
            else:
                raise

    except ImportError as e:
        print(f"JAX not available for FFT: {e}", flush=True)
        raise ImportError("JAX is required for FFT operations")

    def fft_partitioner(fft_func: Callable[[Any], Any], partition_spec: P):
        """Create partitioned FFT function for distributed computation."""
        @custom_partitioning
        def func(x):
            return fft_func(x)

        def partition(mesh, arg_shapes, result_shape):
            mesh = jax.tree.map(lambda x: x.sharding, arg_shapes)[0].mesh
            namedsharding = NamedSharding(mesh, partition_spec)
            return mesh, fft_func, namedsharding, (namedsharding,)

        def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
            mesh = jax.tree.map(lambda x: x.sharding, arg_shapes)[0].mesh
            return NamedSharding(mesh, partition_spec)

        func.def_partition(partition=partition,
                            infer_sharding_from_operands=infer_sharding_from_operands
        )
        return func

    # Define JAX FFT functions (now that JAX is imported and configured)
    def _fft_XY(x):
        return jax.numpy.fft.fftn(x, axes=[0, 1])

    def _fft_Z(x):
        return jax.numpy.fft.rfft(x, axis=2)

    def _ifft_XY(x):
        return jax.numpy.fft.ifftn(x, axes=[0, 1])

    def _ifft_Z(x):
        return jax.numpy.fft.irfft(x, axis=2)

    # CORRECTED sharding patterns based on working implementation
    fft_XY = fft_partitioner(_fft_XY, P(None, None, "gpus"))  # Keep this for XY
    fft_Z = fft_partitioner(_fft_Z, P(None, "gpus"))         # FIXED: Z-axis should be P(None, "gpus")
    ifft_XY = fft_partitioner(_ifft_XY, P(None, None, "gpus"))
    ifft_Z = fft_partitioner(_ifft_Z, P(None, "gpus"))       # FIXED: Z-axis should be P(None, "gpus")

    # CORRECTED order based on working implementation
    def rfftn(x):
        x = fft_Z(x)   # Z first
        x = fft_XY(x)  # then XY
        return x

    def irfftn(x):
        x = ifft_XY(x)  # XY first  
        x = ifft_Z(x)   # then Z
        return x
    
    # Distributed FFT implementation based on working reference
    if jax.process_count() > 1:
        print(f"DISTRIBUTED MODE: Setting up sharded FFT with {jax.process_count()} processes", flush=True)
        
        # Calculate global shape: each process has a Y-slab, combine them
        num_gpus = jax.process_count()
        global_shape = (x_np.shape[0], x_np.shape[1] * num_gpus, x_np.shape[2])
        print(f"Local slab shape: {x_np.shape}, Global shape: {global_shape}", flush=True)
        
        # Create mesh and sharding
        devices = mesh_utils.create_device_mesh((num_gpus,))
        mesh = Mesh(devices, axis_names=('gpus',))
        print(f"JAX mesh created: {mesh}", flush=True)
        
        with mesh:
            print(f"Converting local slab to JAX array (process {jax.process_index()})", flush=True)
            x_single = jax.device_put(x_np).block_until_ready()
            del x_np ; gc.collect()
            
            print(f"Creating sharded array from single-device arrays", flush=True)
            # CRITICAL: Use make_array_from_single_device_arrays for proper distribution
            xshard = jax.make_array_from_single_device_arrays(
                global_shape,
                NamedSharding(mesh, P(None, "gpus")),  # Shard along Y-axis (2nd dimension)
                [x_single]
            ).block_until_ready()
            del x_single ; gc.collect()
            print(f"Sharded array created: {xshard.sharding}", flush=True)
            
            # Set up JIT compilation with explicit input and output shardings
            if direction == 'r2c':
                print(f"Compiling sharded rFFT", flush=True)
                rfftn_jit = jax.jit(
                    rfftn,
                    in_shardings=NamedSharding(mesh, P(None, "gpus")),
                    out_shardings=NamedSharding(mesh, P(None, "gpus"))
                )
            else:
                print(f"Compiling sharded irFFT", flush=True)
                irfftn_jit = jax.jit(
                    irfftn,
                    in_shardings=NamedSharding(mesh, P(None, "gpus")),
                    out_shardings=NamedSharding(mesh, P(None, "gpus"))
                )
            
            from jax.experimental.multihost_utils import sync_global_devices
            sync_global_devices("wait for compiler output")
            
            print(f"Starting sharded FFT computation (process {jax.process_index()}/{jax.process_count()})", flush=True)
            with jax.spmd_mode('allow_all'):
                if direction == 'r2c':
                    out_jit = rfftn_jit(xshard).block_until_ready()
                else:
                    out_jit = irfftn_jit(xshard).block_until_ready()
                sync_global_devices("FFT computation complete")
                
                # Extract local result - this should be the local slab, not the full global array
                local_result = out_jit.addressable_data(0)
                print(f"Sharded FFT complete, local result shape: {local_result.shape}", flush=True)
                
                # Verify the local result has the expected slab shape
                local_slab_shape = (global_shape[0], global_shape[1] // num_gpus, global_shape[2])
                expected_local_shape = (local_slab_shape[0], local_slab_shape[1], local_slab_shape[2]//2 + 1)  # rFFT shape
                if local_result.shape != expected_local_shape:
                    raise RuntimeError(f"FFT output shape {local_result.shape} != expected slab shape {expected_local_shape}. "
                                     f"JAX distributed FFT failed to return correct local slab.")
                
        return np.array(local_result)
    
    else:
        # Single process mode - simple implementation
        print(f"SINGLE PROCESS MODE: Using standard JAX FFT", flush=True)
        x_jax = jax.device_put(x_np).block_until_ready()
        del x_np ; gc.collect()
        
        if direction == 'r2c':
            result = jax.numpy.fft.rfftn(x_jax)
        else:
            result = jax.numpy.fft.irfftn(x_jax)
        
        print(f"Single-process FFT complete", flush=True)
        return np.array(result)