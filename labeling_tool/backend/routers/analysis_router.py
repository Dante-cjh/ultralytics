from fastapi import APIRouter, HTTPException

from .. import schemas
from ..ballistic_analysis import run_ballistic_analysis

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/ballistic", response_model=schemas.BallisticAnalysisResponse)
def ballistic_analysis(body: schemas.BallisticAnalysisRequest):
    if body.hole_areas_px is not None and len(body.hole_areas_px) != len(body.centers_px):
        raise HTTPException(
            status_code=400,
            detail="hole_areas_px 与 centers_px 条数不一致",
        )
    try:
        result = run_ballistic_analysis(
            centers_px=[(p[0], p[1]) for p in body.centers_px],
            pixel_width=body.pixel_width,
            pixel_height=body.pixel_height,
            real_width_m=body.real_width_m,
            real_height_m=body.real_height_m,
            distance_m=body.distance_m,
            is_vertical=body.is_vertical,
            baseline_px=body.baseline_px,
            hole_areas_px=body.hole_areas_px,
        )
        return schemas.BallisticAnalysisResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
