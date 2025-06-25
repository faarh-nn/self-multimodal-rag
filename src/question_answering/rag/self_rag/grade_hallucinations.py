from pydantic import BaseModel as PydanticBaseModel, Field


class GradeHallucinations(PydanticBaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Jawaban ini didasarkan pada konteks-konteks yang ada?, 'yes' or 'no'"
    )