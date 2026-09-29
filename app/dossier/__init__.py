__all__ = [
    "run_dossier_for_peer_set",
    "resume_dossier_run",
    "open_dossier_run",
    "compare_dossier_metric",
]


def __getattr__(name: str):
    if name in __all__:
        from app.dossier.runner import (
            compare_dossier_metric,
            open_dossier_run,
            resume_dossier_run,
            run_dossier_for_peer_set,
        )

        exports = {
            "run_dossier_for_peer_set": run_dossier_for_peer_set,
            "resume_dossier_run": resume_dossier_run,
            "open_dossier_run": open_dossier_run,
            "compare_dossier_metric": compare_dossier_metric,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
