# Training Speed & Memory Optimization Specification

## 1. Overview
This track defines the architectural and data engineering optimizations proposed to increase training throughput, reduce memory footprints, and eliminate RAM bottlenecks on consumer accelerators (macOS Apple Silicon MPS and NVIDIA CUDA).

---

## 2. High-Impact Optimization Targets

### A. Scaled Dot Product Attention (SDPA) Fused Causal Kernel
*   **Description:** The default custom attention mask in [model_tiny_gpt.py](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py) prevents the execution of PyTorch's native fused attention kernel, resulting in large intermediate matrix allocations.
*   **Optimization:** Implement a fast-path that detects when no segment mask is active (`sep_mask_enabled: false`), passing `attn_mask=None` and `is_causal=True` to `scaled_dot_product_attention` to invoke the native fused causal kernel.

### B. Grouped-Query Attention (GQA) via `n_kv_head`
*   **Description:** Multi-Head Attention (MHA) projects and stores key-value (KV) activations for every head, leading to significant memory overhead in deep models.
*   **Optimization:** Implement support for GQA by adding an `n_kv_head` parameter to configs. This projects fewer KV heads than query heads, reducing KV cache and activation memory.

### C. Lazy & Memory-Mapped Dataset Loader
*   **Description:** Preloading the entire dataset into RAM by converting dynamic integer arrays to PyTorch tensors causes high RAM pressure and long startup times.
*   **Optimization:** Implement memory-mapped file loading (`np.load(..., mmap_mode="r")`) to stream sequences from disk, storing file offsets and casting indices to tensors only when batching.

### D. Sequence Length Bucketing / Collation
*   **Description:** Random shuffling of sequences of different lengths forces the collator to pad batches to the maximum sequence length, wasting computation on padding tokens.
*   **Optimization:** Group sequences into buckets of similar lengths, and sample batches from within each bucket to minimize padding.

### E. Block Size Scaling (Sequence Length)
*   **Description:** Attention memory scales quadratically ($O(T^2)$) with the sequence block size.
*   **Optimization:** Reduce the default context window (e.g. from 512 to 384 or 320) in configs to achieve a 40–60% reduction in attention memory.

### F. DataLoader Prefetching & Multi-Processing
*   **Description:** Single-threaded loading (`num_workers: 0`) starves the accelerator during data-heavy training steps.
*   **Optimization:** Configure `num_workers > 0` with `persistent_workers=True` and `prefetch_factor` enabled, while keeping `pin_memory=False` on Apple Silicon.

### G. Prioritized CUDA Device Selection
*   **Description:** The device selection helper defaults to MPS, otherwise falling back to CPU, which ignores CUDA availability on NVIDIA environments.
*   **Optimization:** Update the `dev()` selection logic to check and prefer `cuda` if available.

---

## 3. Metrics & Validation
*   **Throughput:** Tokens processed per second.
*   **Memory Footprint:** Peak GPU and CPU memory usage.
*   **Correctness:** Ensure validation loss and perplexity remain identical to baseline configurations when using mathematically equivalent optimizations.
