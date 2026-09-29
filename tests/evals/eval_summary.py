from __future__ import annotations

import json

from tests.evals.eval_utils import build_eval_summary


def main() -> None:
    print(json.dumps(build_eval_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
