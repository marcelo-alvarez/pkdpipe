from typing import Callable, Any
import gc
import os
import numpy as np

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
    
    # CRITICAL: Initialize JAX distributed mode exactly once, here at FFT time
    # This is the ONLY place where JAX should be imported and initialized
    try:
        # Import JAX modules
        import jax
        from jax import jit
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.experimental.custom_partitioning import custom_partitioning
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        
        # Initialize JAX distributed mode if needed
        slurm_ntasks = os.environ.get('SLURM_NTASKS')
        if slurm_ntasks and int(slurm_ntasks) > 1:
            try:
                # Check if JAX distributed is already initialized
                if jax.process_count() == 1:
                    print("Initializing JAX distributed mode in fft()...", flush=True)
                    
                    # Initialize JAX distributed mode
                    import jax.distributed
                    coordinator_address = os.environ.get('SLURM_STEP_NODELIST', 'localhost').split(',')[0]
                    # Clean up SLURM nodelist format
                    if '[' in coordinator_address:
                        coordinator_address = coordinator_address.split('[')[0] + coordinator_address.split('[')[1].split('-')[0].replace(']', '')
                    
                    jax.distributed.initialize(
                        coordinator_address=f"{coordinator_address}:63025",
                        num_processes=int(slurm_ntasks),
                        process_id=int(os.environ.get('SLURM_PROCID', 0))
                    )
                    print(f"JAX distributed initialized: process {jax.process_index()}/{jax.process_count()}", flush=True)
                else:
                    print(f"JAX distributed already initialized: process {jax.process_index()}/{jax.process_count()}", flush=True)
            except Exception as e:
                print(f"JAX distributed initialization failed, continuing with single process: {e}", flush=True)
        
        # Set JAX configuration
        jax.config.update("jax_enable_x64", False)  # Use 32-bit precision
        jax.config.update("jax_platform_name", "gpu")  # Force GPU platform
        
    except ImportError as e:
        print(f"JAX not available for FFT: {e}", flush=True)
        raise ImportError("JAX is required for FFT operations")
    
    # Define JAX FFT functions (now that JAX is imported)
    def _fft_XY(x):
        return jax.numpy.fft.fftn(x, axes=[0, 1])

    def _fft_Z(x):
        return jax.numpy.fft.rfft(x, axis=2)

    def _ifft_XY(x):
        return jax.numpy.fft.ifftn(x, axes=[0, 1])

    def _ifft_Z(x):
        return jax.numpy.fft.irfft(x, axis=2)

    def fft_partitioner(fft_func: Callable[[Any], Any], partition_spec: Any):
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

        func.def_partition(partition, infer_sharding_from_operands)
        return func

    fft_XY = fft_partitioner(_fft_XY, P(None, None, "gpus"))
    fft_Z = fft_partitioner(_fft_Z, P(None, None, "gpus"))
    ifft_XY = fft_partitioner(_ifft_XY, P(None, None, "gpus"))
    ifft_Z = fft_partitioner(_ifft_Z, P(None, None, "gpus"))

    rfftn = lambda x: fft_Z(fft_XY(x))
    irfftn = lambda x: ifft_XY(ifft_Z(x))
    
    num_gpus = jax.device_count()

    global_shape = (x_np.shape[0], x_np.shape[1]*num_gpus, x_np.shape[2])

    devices = mesh_utils.create_device_mesh((num_gpus,))
    mesh = Mesh(devices, axis_names=('gpus',))
    with mesh:
        print(f"CRITICAL MEMORY TRANSFER: Converting NumPy density grid {x_np.shape} {x_np.dtype} from CPU to JAX arrays on GPU", flush=True)
        print(f"Memory transfer size: {x_np.nbytes / (1024**3):.2f} GB", flush=True)
        
        # *** CRITICAL POINT: CPU NumPy → GPU JAX conversion happens HERE ***
        # x_np is the density grid computed entirely on CPU using multiprocessing
        # jax.device_put() transfers it to GPU memory as JAX arrays
        x_single = jax.device_put(x_np).block_until_ready()
        print(f"NumPy→JAX conversion complete, freeing CPU array", flush=True)
        del x_np ; gc.collect()  # Free CPU memory immediately
        
        xshard = jax.make_array_from_single_device_arrays(
            global_shape,
            NamedSharding(mesh, P(None, "gpus")),
            [x_single]).block_until_ready()
        del x_single ; gc.collect()
        
        if direction=='r2c':
            rfftn_jit = jit(
                rfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
        else:
            irfftn_jit = jit(
                irfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
        sync_global_devices("wait for compiler output")

        with jax.spmd_mode('allow_all'):
            print(f"Starting distributed JAX FFT computation...", flush=True)
            if direction=='r2c':
                out_jit = rfftn_jit(xshard).block_until_ready()
            else:
                out_jit = irfftn_jit(xshard).block_until_ready()
            sync_global_devices("loop")
            
            print(f"JAX FFT complete, converting result back to NumPy", flush=True)
            local_out_subset = out_jit.addressable_data(0)
            
    # Convert back to NumPy for compatibility with downstream processing
    return np.array(local_out_subset)