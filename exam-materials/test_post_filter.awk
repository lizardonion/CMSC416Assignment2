#!/usr/bin/awk -f

# filter some additional lines of output during tests; mostly warnings
# on Zaratan

/Warning: UCX is unable to handle VM_UNMAP event/{next}      
/hwloc x86 backend cannot work under Valgrind, disabling/{next}
/May be reenabled by dumping CPUIDs with hwloc-gather-cpuid/{next}
/and reloading them under Valgrind with HWLOC_CPUID_PATH/{next}
{print}                         # print if not otherwise
