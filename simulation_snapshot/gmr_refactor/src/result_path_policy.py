from .common import sha256_json


def build_result_component(
    verbose_component,
    *,
    compact,
    run_tag,
    feedback_tag,
    schedule_hash="",
    fidelity_hash="",
    seed=None,
):
    if not verbose_component:
        raise ValueError("verbose_component must be nonempty")
    if not compact:
        return verbose_component
    digest = sha256_json({
        "version": 1,
        "marker": "stage_d3_compact_result_path_v1",
        "verbose_component": verbose_component,
        "run_tag": run_tag,
        "feedback_tag": feedback_tag,
        "schedule_hash": schedule_hash,
        "fidelity_hash": fidelity_hash,
        "seed": seed,
    })[:20]
    return f"{run_tag}__{feedback_tag}__compact_{digest}"
