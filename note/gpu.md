# HPC on Oscar Cluster

## Priority Account 
`lgouskos-h100-gcondo` has 1 node with 4 `H100` GPUs and 128 CPUs with 1.5 T RAM.

`lgouskos-l40s-gcondo` has some `L40s` GPUs.
- One can check the available CPUs and GPUs by 
    ```
    sinfo -p $CONDO_NAME -o "%N %c %C %m"
    sinfo -p $CONDO_NAME -o "%P %G %D"
    ```
In conclusion, I have access to **4 H100 GPUs and 128 CPUS**

To check if there is anyone using the condo: 
```
squeue -A $CONDO_NAME -o "%.18i %.8u %.9P %.2t %b"
```

## Usage Checking
Use ```top``` to check job running status. To focus on one user, press `u` and enter username.

| Column   | Meaning |
|----------|---------|
| **PID**  | **Process ID** – the unique identifier for the running process. Every process in Linux has a PID. |
| **USER** | The username of the process owner. Shows who started the process. |
| **PR**   | **Priority** of the process. Lower numbers = higher priority. Linux uses a numeric scale to schedule CPU time. |
| **NI**   | **Nice value** – a user-controlled adjustment to priority. Positive values lower priority, negative values increase it. |
| **VIRT** | **Virtual memory size** of the process. Includes all memory the process can access: code, data, shared libraries, and swap. |
| **RES**  | **Resident memory** – the portion of memory currently in RAM (not swapped out). This is what the process is actively using in physical memory. |
| **SHR**  | **Shared memory** – memory shared with other processes (e.g., shared libraries). |
| **S**    | **Process state** – shows what the process is doing: <br> `R` = running, `S` = sleeping, `D` = uninterruptible sleep, `Z` = zombie, `T` = stopped. |
| **%CPU** | Percentage of **CPU time** used by this process. Can exceed 100% on multi-core systems because it sums over all cores. |
| **%MEM** | Percentage of **physical RAM** used by the process (based on RES / total RAM). |
| **TIME+**| Total **CPU time** the process has used since it started (user + system time), shown as `minutes:seconds.tenths`. |
| **COMMAND** | The **name or command** of the process (e.g., `python`, `bash`, `java`). |
