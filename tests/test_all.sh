#!/bin/bash
# ===----------------------------------------------------------------------=== #
# Run all Stamojo tests
# ===----------------------------------------------------------------------=== #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running Stamojo tests ==="
echo ""

# Add test commands here as they are developed
# pixi run mojo test -I "$SCRIPT_DIR" "$SCRIPT_DIR/test_distributions.mojo"
# pixi run mojo test -I "$SCRIPT_DIR" "$SCRIPT_DIR/test_stats.mojo"
# pixi run mojo test -I "$SCRIPT_DIR" "$SCRIPT_DIR/test_models.mojo"

echo "No tests yet. Tests will be added as modules are implemented."
echo ""
echo "=== All tests completed ==="
