from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Applicant(Base):
    __tablename__ = "applicants"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    applicant_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    gender: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    education: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    self_employed: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    no_of_dependents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    income_annum: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    loan_amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    loan_term: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    cibil_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    
    prediction: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
    )