from typing import Callable, Any
import gc
import os
import numpy as np

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
    JAX-based distributed FFT with enhanced hang prevention and coordination.
    
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
        # Initialize JAX distributed mode BEFORE any other JAX operations
        process_id = int(os.environ.get('SLURM_PROCID', 0))
        print(f"Process {process_id}: Initializing JAX distributed mode in fft()...", flush=True)
        
        # CRITICAL FIX: Enhanced MPI coordination before JAX initialization
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            # Add timeout detection for MPI barrier itself
            import time
            import signal
            import threading
            
            def barrier_timeout_handler(signum, frame):
                raise TimeoutError("MPI barrier timeout in FFT initialization")
            
            print(f"Process {process_id}: MPI rank {rank} of {size} - waiting at barrier before JAX init", flush=True)
            
            # Set 60-second timeout for MPI barrier
            signal.signal(signal.SIGALRM, barrier_timeout_handler)
            signal.alarm(60)
            
            try:
                # Enhanced barrier with progress monitoring
                barrier_start = time.time()
                barrier_success = threading.Event()
                barrier_error = [None]
                
                def do_barrier():
                    try:
                        comm.Barrier()
                        barrier_success.set()
                    except Exception as e:
                        barrier_error[0] = e
                        barrier_success.set()
                
                # Start barrier in thread with timeout monitoring
                barrier_thread = threading.Thread(target=do_barrier, daemon=True)
                barrier_thread.start()
                
                # Monitor barrier progress
                timeout_seconds = 60
                check_interval = 5
                for i in range(0, timeout_seconds, check_interval):
                    if barrier_success.wait(timeout=check_interval):
                        break
                    elapsed = time.time() - barrier_start
                    print(f"Process {process_id}: MPI barrier still waiting after {elapsed:.1f}s", flush=True)
                else:
                    # Barrier timed out
                    elapsed = time.time() - barrier_start
                    print(f"Process {process_id}: MPI barrier timed out after {elapsed:.1f}s", flush=True)
                    raise TimeoutError(f"MPI barrier hung for {elapsed:.1f}s in FFT initialization")
                
                if barrier_error[0]:
                    raise barrier_error[0]
                    
                signal.alarm(0)  # Cancel timeout
                barrier_elapsed = time.time() - barrier_start
                print(f"Process {process_id}: MPI barrier complete ({barrier_elapsed:.1f}s) - proceeding to JAX init", flush=True)
                
            except TimeoutError as te:
                signal.alarm(0)
                print(f"Process {process_id}: MPI barrier timeout: {te}", flush=True)
                # Don't fail immediately - try to proceed with JAX init but log the issue
                print(f"Process {process_id}: WARNING: Proceeding with JAX init despite barrier timeout", flush=True)
            
        except Exception as mpi_e:
            print(f"Process {process_id}: MPI coordination failed: {mpi_e}, proceeding without barrier", flush=True)
        
        try:
            # CRITICAL: Import jax first, then jax.distributed
            import jax
            import jax.distributed
            
            # Enhanced SLURM environment debugging
            print(f"Process {process_id}: SLURM_NTASKS={slurm_ntasks}", flush=True)
            print(f"Process {process_id}: SLURM_PROCID={process_id}", flush=True)
            
            # Get coordinator from SLURM nodelist with better error handling
            nodelist = os.environ.get('SLURM_STEP_NODELIST', 'localhost')
            print(f"Process {process_id}: Raw SLURM_STEP_NODELIST: {nodelist}", flush=True)
            
            # IMPROVED: Better coordinator address parsing for multi-node jobs
            coordinator_address = nodelist.split(',')[0] if ',' in nodelist else nodelist
            
            # Clean up SLURM nodelist format more carefully  
            if '[' in coordinator_address:
                base_name = coordinator_address.split('[')[0]
                range_part = coordinator_address.split('[')[1].split('-')[0].replace(']', '')
                coordinator_address = base_name + range_part
            
            # FALLBACK: Try alternate SLURM environment variables
            if not coordinator_address or coordinator_address == 'localhost':
                alt_nodelist = os.environ.get('SLURM_NODELIST', 'localhost')
                print(f"Process {process_id}: Trying SLURM_NODELIST: {alt_nodelist}", flush=True)
                if alt_nodelist != 'localhost':
                    coordinator_address = alt_nodelist.split(',')[0] if ',' in alt_nodelist else alt_nodelist
                    if '[' in coordinator_address:
                        base_name = coordinator_address.split('[')[0]
                        range_part = coordinator_address.split('[')[1].split('-')[0].replace(']', '')
                        coordinator_address = base_name + range_part
                        
            # Validate coordinator address
            if not coordinator_address or coordinator_address == 'localhost':
                print(f"Process {process_id}: WARNING: Using localhost as coordinator", flush=True)
                
            print(f"Process {process_id}: Parsed coordinator address: {coordinator_address}", flush=True)
            print(f"Process {process_id}: JAX distributed processes: {slurm_ntasks}, process_id: {process_id}", flush=True)
            
            # Initialize distributed mode with enhanced timeout detection
            print(f"Process {process_id}: Calling jax.distributed.initialize...", flush=True)
            
            # ENHANCED: Try with explicit timeout and error handling
            import signal
            import time
            import threading
            
            def jax_init_timeout_handler(signum, frame):
                raise TimeoutError("JAX distributed initialization timeout")
                
            # Set 45-second timeout for initialization (reduced for faster failure detection)
            signal.signal(signal.SIGALRM, jax_init_timeout_handler)
            signal.alarm(45)
            
            try:
                # Enhanced heartbeat monitoring during JAX initialization
                heartbeat_stop = threading.Event()
                
                def heartbeat_monitor():
                    count = 0
                    while not heartbeat_stop.is_set():
                        time.sleep(5)
                        count += 1
                        print(f"Process {process_id}: JAX init heartbeat {count*5}s", flush=True)
                
                monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
                monitor_thread.start()
                
                jax.distributed.initialize(
                    coordinator_address=f"{coordinator_address}:63025",
                    num_processes=int(slurm_ntasks),
                    process_id=process_id,
                    local_device_ids=None  # Let JAX auto-detect devices
                )
                
                heartbeat_stop.set()
                signal.alarm(0)  # Cancel timeout
                print(f"Process {process_id}: JAX distributed initialized successfully", flush=True)
                
            except TimeoutError:
                heartbeat_stop.set()
                signal.alarm(0)
                print(f"Process {process_id}: JAX distributed initialization timed out after 45s", flush=True)
                raise RuntimeError(f"Process {process_id}: JAX distributed initialization timeout - likely hanging")
            
        except Exception as e:
            print(f"Process {process_id}: JAX distributed initialization failed: {e}", flush=True)
            print(f"Process {process_id}: Error type: {type(e).__name__}", flush=True)
            print(f"Process {process_id}: Coordinator: {coordinator_address}:63025", flush=True)
            print(f"Process {process_id}: Num processes: {slurm_ntasks}, Process ID: {process_id}", flush=True)
            
            # NEW: Try fallback to single-process mode if distributed fails
            print(f"Process {process_id}: FALLBACK: Attempting single-process JAX mode", flush=True)
            is_distributed = False
            try:
                import jax
                print(f"Process {process_id}: JAX imported in single-process fallback mode", flush=True)
            except Exception as fallback_e:
                raise RuntimeError(f"Process {process_id}: Both distributed and single-process JAX failed") from e
    
    # NOW safe to import other JAX modules after distributed mode is set up
    try:
        # JAX already imported in distributed block if needed, check if we need to import it
        if 'jax' not in locals():
            import jax
            
        # Get process ID for logging (may not be available if not distributed)
        process_id = int(os.environ.get('SLURM_PROCID', 0)) if is_distributed else 0
        print(f"Process {process_id}: Importing JAX modules after distributed setup...", flush=True)
        
        from jax import jit
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.experimental.custom_partitioning import custom_partitioning
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
        
        print(f"Process {process_id}: JAX modules imported successfully", flush=True)
        
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
    
    # Distributed FFT implementation with enhanced hang prevention
    if is_distributed and jax.process_count() > 1:
        print(f"DISTRIBUTED MODE: Setting up sharded FFT with {jax.process_count()} processes", flush=True)
        
        # ENHANCED CHECKPOINT 1: Pre-FFT synchronization with timeout
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            print(f"[FFT] Rank {rank}: Entering distributed FFT section", flush=True)
            
            # Add timeout for this barrier too
            import signal
            import time
            import threading
            
            def fft_barrier_timeout_handler(signum, frame):
                raise TimeoutError("MPI barrier timeout in FFT setup")
            
            signal.signal(signal.SIGALRM, fft_barrier_timeout_handler)
            signal.alarm(30)  # 30 second timeout for FFT setup barrier
            
            try:
                barrier_start = time.time()
                comm.Barrier()
                barrier_elapsed = time.time() - barrier_start
                signal.alarm(0)
                print(f"[FFT] Rank {rank}: All ranks synchronized before FFT setup ({barrier_elapsed:.1f}s)", flush=True)
            except TimeoutError:
                signal.alarm(0)
                print(f"[FFT] Rank {rank}: FFT setup barrier timed out - proceeding anyway", flush=True)
                
        except Exception as e:
            print(f"[FFT] MPI sync failed: {e}", flush=True)
        
        # Calculate global shape: each process has a Y-slab, combine them
        num_gpus = jax.process_count()
        global_shape = (x_np.shape[0], x_np.shape[1] * num_gpus, x_np.shape[2])
        print(f"Local slab shape: {x_np.shape}, Global shape: {global_shape}", flush=True)
        
        # Create mesh and sharding with timeout monitoring
        print(f"[FFT] Rank {rank}: Creating device mesh...", flush=True)
        devices = mesh_utils.create_device_mesh((num_gpus,))
        mesh = Mesh(devices, axis_names=('gpus',))
        print(f"JAX mesh created: {mesh}", flush=True)
        
        # ENHANCED CHECKPOINT 2: Post-mesh creation synchronization with timeout
        try:
            signal.signal(signal.SIGALRM, fft_barrier_timeout_handler)
            signal.alarm(30)
            
            try:
                barrier_start = time.time()
                comm.Barrier()
                barrier_elapsed = time.time() - barrier_start
                signal.alarm(0)
                print(f"[FFT] Rank {rank}: All ranks synchronized after mesh creation ({barrier_elapsed:.1f}s)", flush=True)
            except TimeoutError:
                signal.alarm(0)
                print(f"[FFT] Rank {rank}: Post-mesh barrier timed out - proceeding anyway", flush=True)
        except:
            pass
        
        with mesh:
            print(f"Converting local slab to JAX array (process {jax.process_index()})", flush=True)
            x_single = jax.device_put(x_np).block_until_ready()
            del x_np ; gc.collect()
            
            print(f"Creating sharded array from single-device arrays", flush=True)
            
            # CRITICAL FIX: Enhanced timeout detection for make_array_from_single_device_arrays
            start_time = time.time()
            print(f"[FFT] Process {jax.process_index()}: Starting make_array_from_single_device_arrays at {start_time:.2f}", flush=True)
            
            # Set up timeout handler for the sharding operation
            sharding_success = threading.Event()
            sharding_result = [None]
            sharding_error = [None]
            
            def do_sharding():
                try:
                    # CRITICAL: Use make_array_from_single_device_arrays for proper distribution
                    result = jax.make_array_from_single_device_arrays(
                        global_shape,
                        NamedSharding(mesh, P(None, "gpus")),  # Shard along Y-axis (2nd dimension)
                        [x_single]
                    ).block_until_ready()
                    sharding_result[0] = result
                    sharding_success.set()
                except Exception as e:
                    sharding_error[0] = e
                    sharding_success.set()
            
            # Start sharding in separate thread with timeout
            sharding_thread = threading.Thread(target=do_sharding, daemon=True)
            sharding_thread.start()
            
            # Wait for sharding to complete with timeout
            if sharding_success.wait(timeout=60):  # Increased to 60 second timeout
                elapsed = time.time() - start_time
                print(f"[FFT] Process {jax.process_index()}: Sharding took {elapsed:.2f}s", flush=True)
                
                if sharding_error[0]:
                    raise sharding_error[0]
                    
                xshard = sharding_result[0]
            else:
                print(f"[FFT] Process {jax.process_index()}: Sharding timed out after 60s - HANGING DETECTED", flush=True)
                raise RuntimeError(f"Process {jax.process_index()}: make_array_from_single_device_arrays hung")
            
            del x_single ; gc.collect()
            print(f"Sharded array created: {xshard.sharding}", flush=True)
            
            # ENHANCED CHECKPOINT 3: Post-sharding synchronization with timeout
            try:
                signal.signal(signal.SIGALRM, fft_barrier_timeout_handler)
                signal.alarm(30)
                
                try:
                    barrier_start = time.time()
                    comm.Barrier()
                    barrier_elapsed = time.time() - barrier_start
                    signal.alarm(0)
                    print(f"[FFT] Rank {rank}: All ranks synchronized after sharding ({barrier_elapsed:.1f}s)", flush=True)
                except TimeoutError:
                    signal.alarm(0)
                    print(f"[FFT] Rank {rank}: Post-sharding barrier timed out - proceeding anyway", flush=True)
            except:
                pass
            
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