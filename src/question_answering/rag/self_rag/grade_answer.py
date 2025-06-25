from pydantic import BaseModel as PydanticBaseModel, Field


class GradeAnswer(PydanticBaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Jawaban ini menjawab pertanyaan?, 'yes' or 'no'"
    )