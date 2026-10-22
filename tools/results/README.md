# Results

The `llama-results` tool can be used to `--check` the outputs of a model vs. a previous commit to detect whether they have changed.
Example usage:

``` sh
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed."  # writes results to file
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed." --check  # compares results vs file
```

The metric by which the results are compared is the normalized mean squared error (NMSE) with a tolerance of $10^{-6}$.

For token-level diagnostics, the tool can also write a JSON top-1 report:

``` sh
llama-results --model model.gguf --prompt "People die when they are killed." --top1-report top1.json
llama-results --model model.gguf --output results.gguf --prompt "People die when they are killed." --check --top1-report top1.json
```

Add `--top1-fail-on-mismatch` to fail the check when any compared position has a different top-1 token.

Use `--sequential-logits` to evaluate the fixed prompt one token at a time. This is slower, but it more closely matches generation-style decoding and avoids large all-token logits batches.
