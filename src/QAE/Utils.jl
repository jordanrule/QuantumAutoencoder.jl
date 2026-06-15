"""
Utility functions for the Julia QAE port.

Where possible, these functions mirror the intent of utilities in QCompress.
"""

export load_dataset

function load_dataset(path::AbstractString, dataset::AbstractString="/data")
    # Try to load using HDF5 if available
    try
        @eval using HDF5
        f = HDF5.h5open(path, "r")
        try
            if haskey(f, dataset)
                data = read(f[dataset])
            else
                # fallback: read first dataset
                keys = collect(keys(f))
                data = isempty(keys) ? nothing : read(f[keys[1]])
            end
        finally
            close(f)
        end
        return data
    catch err
        # Provide a clear error if HDF5 is not installed or file can't be read
        throw(ErrorException("Failed to load HDF5 dataset: $(err). Install HDF5.jl or provide HDF5 data."))
    end
end


