from fastapi import APIRouter


def assert_demo_safe_status() -> dict[str, str]:
    return {
        "mode": "demo",
        "boundary": (
            "Decision-support prototype. Synthetic data is not field-validated "
            "and must not be used for official reserves, blasting, dispatch, or safety decisions."
        ),
    }


router = APIRouter(tags=["settings"])


@router.get("/settings/safety")
def get_safety_settings() -> dict[str, str]:
    return assert_demo_safe_status()

