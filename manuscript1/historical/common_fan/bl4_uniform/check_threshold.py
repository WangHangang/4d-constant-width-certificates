#!/usr/bin/env python3
"""Exit 0 exactly when a generated certificate records a strict T crossing."""
import json, sys
data=json.load(open(sys.argv[1],encoding="utf-8"))
crossed=bool(data.get("strict_threshold_crossed"))
print("ABOVE_THRESHOLD" if crossed else "BELOW_THRESHOLD")
raise SystemExit(0 if crossed else 1)

