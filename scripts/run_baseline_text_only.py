#!/usr/bin/env python
"""Compatibility wrapper around `scripts/run.py baseline --baseline text_only_encoder`."""

from __future__ import annotations

import sys

from run import main


if __name__ == "__main__":
    sys.argv[1:1] = ['baseline', '--baseline', 'text_only_encoder']
    main()
