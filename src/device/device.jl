"""
    AbstractDevice

Supertype of the compute devices on which Hamiltonians can be built, diagonalized and differentiated.
"""
abstract type AbstractDevice end

"""
    CPUDevice()

The default device. All computations use the standard CPU code paths.
"""
struct CPUDevice <: AbstractDevice end

"""
    ArrayDevice(array_type)

A device whose data lives in arrays created by `array_type` (e.g. `CuArray`). Computations run through
the device code path (see `DeviceLinearMap`), which is written with KernelAbstractions and therefore works
for any array type with a KernelAbstractions backend (`Array`, `CuArray`, `JLArray`, ...).
"""
struct ArrayDevice{A} <: AbstractDevice
    array_type :: A
end

to_device(::CPUDevice, x) = x
to_device(dev::ArrayDevice, x::AbstractArray) = dev.array_type(x)

uses_device_path(::AbstractDevice) = false
uses_device_path(::ArrayDevice) = true

"""
    gpu_device(local_rank)

Select a GPU for the given node-local MPI rank and return the corresponding `ArrayDevice`.
Implemented by the `HamsterCUDAExt` extension, which is loaded with `using CUDA`.
"""
function gpu_device end

"""
    gpu_count()

Number of GPUs visible on this node. Implemented by the `HamsterCUDAExt` extension, which is loaded with `using CUDA`.
"""
function gpu_count end

"""
    device_synchronize(x)

Wait until all device operations on the array `x` have finished (no-op on the host). Used before timings are taken,
since device kernels run asynchronously. Device-specific methods are provided by extensions.
"""
device_synchronize(x) = nothing

"""
    shift_invert_operator(A)

Factorize the sparse device matrix `A` and return a function `x -> A \\ x`.
Implemented by the `HamsterCUDSSExt` extension, which is loaded with `using CUDA, CUDSS`.
"""
function shift_invert_operator end

"""
    release_operator!(op)

Free the device memory of an operator from `shift_invert_operator` (e.g. the cuDSS factorization) right away instead
of waiting for the garbage collector.
"""
release_operator!(op) = nothing

"""
    init_device(conf, comm=nothing; device=get_device(conf))

Create the compute device requested by the `device` tag. For `gpu`, each MPI rank is bound to a GPU
of its node in a round-robin fashion based on its node-local rank.
"""
function init_device(conf=get_empty_config(), comm=nothing; device=get_device(conf))
    device = lowercase(device)
    if device == "cpu"
        return CPUDevice()
    elseif device == "gpu"
        if isempty(methods(gpu_device))
            error("device = gpu requires the CUDA extension. Install CUDA.jl and load it with `using CUDA` before running Hamster.")
        end
        return gpu_device(get_node_local_rank(comm))
    else
        error("Unknown device '$device'. Valid options are cpu and gpu.")
    end
end

"""
    get_num_gpus(conf, comm, hostname; device=get_device(conf))

Total number of GPUs used by all MPI ranks, i.e., the number of distinct (node, GPU) pairs that the ranks bind to in `init_device`.
Returns 0 for `device = cpu`. Must be called on all ranks; the result is only valid on rank 0.
"""
function get_num_gpus(conf, comm, hostname; device=get_device(conf))
    lowercase(device) == "gpu" && !isempty(methods(gpu_count)) || return 0
    ngpus_node = gpu_count()
    gpu = ngpus_node > 0 ? (hostname, mod(get_node_local_rank(comm), ngpus_node)) : nothing
    gpus = MPI.gather(gpu, comm, root=0)
    return isnothing(gpus) ? 0 : length(unique(filter(!isnothing, gpus)))
end

function get_node_local_rank(comm)
    isnothing(comm) && return 0
    node_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    local_rank = MPI.Comm_rank(node_comm)
    MPI.free(node_comm)
    return local_rank
end
