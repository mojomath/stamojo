#!/bin/bash
# ===----------------------------------------------------------------------=== #
# Run all Stamojo tests
# ===----------------------------------------------------------------------=== #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running StaMojo tests ==="
echo ""

# Special functions
echo "--- Special functions ---"
pixi run --environment test mojo run -I src "$SCRIPT_DIR/test_special.mojo"
echo ""

# Distribution tests
echo "--- Distributions ---"
pixi run --environment test mojo run -I src "$SCRIPT_DIR/test_distributions.mojo"
echo ""

# Stats tests
echo "--- Descriptive statistics ---"
pixi run --environment test mojo run -I src "$SCRIPT_DIR/test_stats.mojo"
echo ""

# Hypothesis tests
echo "--- Hypothesis tests ---"
pixi run --environment test mojo run -I src "$SCRIPT_DIR/test_hypothesis.mojo"
echo ""

echo "=== All tests completed ==="
