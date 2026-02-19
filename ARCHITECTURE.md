# Architecture: shrew-core

`shrew-core` provides the fundamental abstractions for the Shrew deep learning framework. It defines the tensor structure, backend interface, and automatic differentiation engine, remaining agnostic to the specific execution hardware (CPU vs GPU).

## Core Concepts

- **Tensor<B>**: The primary data structure, parameterized by a `Backend` type `B`. It manages the underlying data storage (via the backend), shape, and autograd metadata.
- **Backend Trait**: An interface that defines how memory is allocated and how operations are executed. Concrete implementations (like `CpuBackend` in `shrew-cpu`) must implement this trait.
- **Autograd**: A reverse-mode automatic differentiation system implemented in `backprop.rs`. It builds a dynamic computation graph during the forward pass and traverses it backward to compute gradients.
- **Dynamic Shapes**: Support for tensors with symbolic dimensions, allowing for flexible model definitions that aren't tied to fixed batch sizes or sequence lengths.

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `tensor.rs` | Defines the `Tensor` struct and its core methods (creation, views, arithmetic dispatch). Central entry point for the API. | 2344 |
| `backprop.rs` | Implements the autograd engine, including `GradStore` for accumulating gradients and the backward pass logic. | 1523 |
| `dynamic_shape.rs` | Handling of dynamic and symbolic shapes, enabling flexible tensor dimensions. | 829 |
| `op.rs` | traits for tensor operations (unary, binary, reduction) that backends must implement. | 576 |
| `layout.rs` | Manages memory layout, strides, and contiguous memory checks. | 283 |
| `backend.rs` | Defines the `Backend` trait, the contract that all execution engines must satisfy. | 232 |
| `shape.rs` | Logic for static tensor shapes and dimension manipulation. | 215 |
| `dtype.rs` | Definitions of supported data types (`f32`, `f64`, etc.) and the `Element` trait. | 185 |
| `error.rs` | Custom error types for shape mismatches, dtype errors, and other runtime failures. | 78 |
| `lib.rs` | Crate root, module exports. | 35 |
