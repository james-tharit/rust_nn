# rust_nn

A tiny neural network in Rust, built for learning. Hand-rolled neurons and layers with a sigmoid activation, plus `ndarray`-based primitives (linear forward, ReLU/sigmoid) for a vectorized path.

## Run

```sh
cargo run
```

Prints the output of a 3-neuron layer over a fixed input vector.

## Test

```sh
cargo test
```

## Layout

- `src/nueron.rs` — single neuron: weighted sum + bias, sigmoid activation
- `src/layer.rs` — layer: fans inputs across its neurons
- `src/activations.rs` — `sigmoid`, `relu`, `linear_forward`, and the `Array2`-based activation wrappers with caches for backprop
- `src/main.rs` — demo
