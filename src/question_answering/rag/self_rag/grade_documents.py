from pydantic import BaseModel as PydanticBaseModel, Field


class GradeDocuments(PydanticBaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Dokumen ini relevan dengan pertanyaan?, 'yes' or 'no'"
    )