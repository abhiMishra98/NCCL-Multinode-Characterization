# NCCL-Multinode-Characterization
Characterizing NCCL Collective Communication Across a Heterogeneous 2-Node GPU Cluster (TCP/Ethernet, with a GPUDirect RDMA Upgrade Path)

## Baseline Performance

- **Baseline network bandwidth:** 942 Mbps (TCP/Ethernet), measured with `iperf3`

## GPU-NIC NUMA Topology

- `lscpu` → check `Socket(s):` to get the number of CPU sockets/NUMA nodes.
- `lspci -tv` → check whether the GPU and NIC hang off the same PCIe root complex.

Example (trimmed):

```
-[0000:00]-+-02.0-[01]----00.0  Mellanox ConnectX-5   <- NIC
           \-1d.0-[3d]----00.0  NVIDIA A100            <- GPU
```

`-[0000:00]-` is the root complex; both the NIC (bus `01`) and GPU (bus `3d`) branch from it, so they're on the same root complex/NUMA node. A device under a different root (e.g. `-[0000:80]-`) would need a cross-socket hop.

PCIe address format — `domain:bus:device.function` (e.g. `0000:3d:00.0`): the `bus` field is what you trace up through the `lspci -tv` tree to find its root complex.

**Result:** on both servers, the GPU and NIC sit under the same root complex — no cross-socket hop for GPUDirect RDMA traffic.
