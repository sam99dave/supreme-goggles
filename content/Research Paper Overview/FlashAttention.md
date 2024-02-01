---
title: FlashAttention, Fast and Memory-Efficient Exact Attention with IO-Awareness
draft: false
tags: 
date: 2024-01-14
---

It takes a lot of time to train LLMs, a transformer based model which uses attention mechanism at its core. Transformers have a reputation of being slow and memory hungry for processing long sequences, their time & memory complexity has a **_quadratic relationship with sequence length_**. We know that attention mechanism is well optimized as compared to LSTM as we can train in parallel whereas we cannot for the latter.

## Approximate Attention Methods

Attempts have been made to tackle this issue by trading model quality for reduction in complexity. Surely complexity is reduced but the expected wall-clock speedup is not witnessed.

These methods reduce the time complexity to linear or near-linear in sequence length which is actually great but this doesn’t reflect in the wall-clock speedup for most of the implementations.

This paper delves into finding the missing piece and with it lays down a baseline for a better implementation of approximate attention methods, `Block-Sparse FlashAttention`.


## The Culprit: IO Overhead

The authors argued that the missing piece is _IO awareness,_ taking into consideration the read & write between the different levels of the GPU.

The approximation approaches focus on reducing the time complexity through FLOP reduction ignoring the overheads from memory access. We will expand on this in the FlashAttention section.

>FlashAttention, an IO aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between the GPU’s high bandwidth memory (HBM) and the GPU’s on-chip SRAM.


## GPU Memory Hierarchy

This hierarchy comprises of multiple forms of memory of different size and speeds, size and speed holding an inverse relationship with each other.

**HBM [ High Bandwidth Memory]**

- Size in range 40–80 GB
- Bandwidth in range 1.5–2 TB/s

**SRAM**

- Consists on around 108 streaming multiprocessors
- Size 192KB per each of the streaming multiprocessors
- Bandwidth around 19 TB/s

The on-chip memory, SRAM is an order of magnitude faster in than HBM but many order of magnitudes smaller in size. Compute speeds have gotten a huge bump as compared to memory speed, therefore, **_operations are getting bottlenecked by memory access_**. Thus, exploiting the SRAM makes so much sense.

**Execution Model**

GPUs have a massive number of threads, these threads are the ones executing the operations ( known as kernels ). Each kernel loads the inputs from the HBM to the register, the SRAM performs the computation and the output is moved back to HBM.

**Performance and Overheads**

An operation can be either compute or memory bound based on the balance of computation and memory access.

_Compute Bound,_ the case where the time taken by the operation is determined by the **_number of arithmetic operations_** other than the time taken to access HBM. _Memory Bound_ is just the opposite of this.

Here are some example operations from both the categories

_Compute Bound_

- Matrix Multiplication with large inner dimension
- Convolution with large number of channels ( dense layers )

_Memory Bound_

- Elementwise Operations like _activation, dropout_
- Reduction Operations like _sum, softmax, batchnorm, layernorm_


**Kernel Fusion**

The most common approach to accelerate memory (IO) bound operations is **_kernel fusion_**.

Kernel Fusion: If there are multiple operations to be applied to the same input, the input is loaded once from the HBM for the fused operation instead of multiple times for each operation (_Compilers can automatically fuse many arithmetic operations_).

In case of model training, the intermediate results still need to be written to HBM to save for the backward pass (hold your thoughts till the recomputation section), this reduces the effectiveness of naive kernel fusion.


## Standard Attention Implementation

N — Sequence Length

d — Head dimension

**Intermediate Results**

_Standard attention implementations materialize the matrices_ **_S_** _&_ **_P_** _to the HBM_, both these matrices have a quadratic complexity relationship **O(N²)** memory where often N >> d.

Eg. for GPT2 , N = 1024 & d = 64

Over here, we have the _softmax operation which if you recall is a memory bound operation_, large number of memory accesses translates to slow wall-clock time. Furthermore, other elementwise operations such as _masking_ applied to S and _dropout_ applied to P adds to this problem. This has encouraged the exploration to fuse the elementwise operations together, such as fusing masking with softmax.

> In the paper, the authors have proved that the standard attention implementation performs HBM accesses quadratic in the sequence length N (i.e. N² times). **section 3.2**


## FlashAttention

The standard attention mechanism stores the intermediates to HBM, FlashAttention on the other hand aims to calculate the exact attention without storing the intermediates thus reducing the read/write overhead on the HBM. This enables both memory efficient & faster wall-clock speed ups.

_Overview:_ From the algorithm figure, we can see that the game plan is to play around blocks,

- Splitting Q,K and V into blocks
- Loading these blocks, one at a time from HBM to SRAM
- Compute the output attention
- Update assets to calculate softmax ( m & l )
- Store it back to the HBM, [ Q(i), m(i) and l(i) ]

Here, as we can see the intermediates are not getting stored in the HBM, the output is being calculated dynamically, output attention block is updated ones in the inner loop & throughout all the values of the outer loop(K & V).

There are few challenges over here in regards to calculating the exact attention without storing the intermediates, S and P. The paper incorporates 2 techniques to overcome this and achieve the result in sub-quadratic HBM accesses. Let’s have a look at them:

**Tiling**

`Tiling enables to implement the algorithm in one CUDA kernel`

Softmax couples columns of K, so we decompose the large softmax with scaling. In the above image, **_x_** is a vector of size ~ no. of blocks.

In the algorithm, m(i) & l(i) are also being tracked and used in softmax calculation.

**Recomputation**

One of the goals of FlashAttention is to not store the O(N²) intermediate values as discussed before but these are needed during the backward pass. Surprisingly, the S & P matrix can be recomputed **_on the SRAM_** using the Q,K,V blocks along with the statistics (m and l) during the backward pass.

> Even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses

And also, **scaling**

>By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end.

## Some Analysis

The middle figure shows the affect of block size on HBM accesses which affects the runtime.

- Large block-size ~ less HBM accesses therefore less runtime

The figure to the right is regarding **_Block-sparse FlashAttention_**, a combination of FlashAttention and approximation methods. It shows the change in time taken by both Forward & Backward pass with increasing sparsity.

- Block-Sparse FlashAttention is faster than FlashAttention by a factor proportional to sparsity.

Furthermore, the authors prove the HBM accesses for both the attention calculation techniques ( Standard & FlashAttention )

- Standard — Θ(𝑁𝑑 + 𝑁²)
- FlashAttention — Θ(N²d²/M), M ~ SRAM size

> Therefore, the authors proved that any standard attention calculation cannot asymptotically improve on HBM accesses over SRAM sizes.


## Block-Sparse FlashAttention

The first component over here is the _sparsity_, achieved with the help of a mask matrix, this renders the values to be,

- −∞ if Mask value is 0
- S(kl), the standard matrix multiplication result if Mask value is 1

The second component is adapting this into FlashAttention, this is done using a **_pre-defined sparsity mask_**:

The IO complexity of Block-Sparse FlashAttention is smaller than that of FlashAttention by a factor proportional to the sparsity.

HBM accesses — Θ(𝑁𝑑 + N²d²s/M). For large value of N, **s** is set to be either 𝑁^(−1/2) or N^(-1) which results in IO complexity (HBM) of Θ(𝑁√ 𝑁) or Θ(𝑁log 𝑁)

## Outcomes

**Faster Training**

- FlashAttention outperforms the MLPerf 1.1 speed record for BERT by 15%
- Speeds up GPT-2 up to 3× over HuggingFace and 1.8× over Megatron over standard Transformers.
- Speeds up the long-range arena (LRA) benchmark by 2.4×.

**Model Quality**

- FlashAttention scales Transformers to longer sequences.
- Modeling longer sequences yields **_6.4 points of lift_** on two long document classification tasks
- FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging Path-X task
- Block-sparse FlashAttention yields the first sequence model that we know of that can achieve better-than-random performance on Path-256

**Attention Benchmarks**

- Memory footprint of FlashAttention scales linearly with seq. length and is up to 3× faster than standard attention for common seq. lengths (up to 2K).
- Block-sparse FlashAttention scales linearly in seq. length and is faster than all **_existing approximate attention baselines._**

## Limitations

**Compiling to CUDA**

- IO aware attention implementation requires writing a new CUDA kernel which is a lot of engineering effort.
- Implementations may also not be transferrable across GPU architectures.

## Future Directions

**IO aware Deep Learning**

Every layer in deep networks interacts with the HBM so there is possible path for IO aware modules implementation for deep learning.

**Multi-GPU IO-aware methods**

The current IO-aware implementation introduced by this paper is optimal with constants for computing attention on a single GPU. Introducing parallelism for attention computation across multiple GPU might be a possibility.





