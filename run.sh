#!/bin/bash

# Scalability results
uv run python examples/test_expressivity.py
uv run python examples/time_expressivity.py
uv run python examples/plot_expressivity.py

# Performance/training time results
uv run python examples/train_observer.py
uv run python examples/train_yoularen.py
uv run python examples/train_sysid.py
uv run python examples/plot_performance_comparison.py
