#!/bin/bash

source venv/bin/activate

# Scalability results
python examples/test_expressivity.py
python examples/time_expressivity.py
python examples/plot_expressivity.py

# Performance/training time results
python examples/train_observer.py
python examples/train_yoularen.py
python examples/train_sysid.py
python examples/plot_performance_comparison.py