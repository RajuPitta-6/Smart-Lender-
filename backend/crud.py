from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Applicant
from backend.schemas import ApplicantCreate


def create_applicant(
    db: Session,
    applicant: ApplicantCreate,
    prediction: str,
    confidence: float,
):
    db_applicant = Applicant(
        applicant_name=applicant.applicant_name,
        gender=applicant.gender,
        education=applicant.education,
        self_employed=applicant.self_employed,
        no_of_dependents=applicant.no_of_dependents,
        income_annum=applicant.income_annum,
        loan_amount=applicant.loan_amount,
        loan_term=applicant.loan_term,
        cibil_score=applicant.cibil_score,
        prediction=prediction,
        confidence=confidence,
    )

    db.add(db_applicant)
    db.commit()
    db.refresh(db_applicant)

    return db_applicant


def get_all_applicants(db: Session):
    statement = select(Applicant)

    return db.scalars(statement).all()


def get_applicant_by_id(
    db: Session,
    applicant_id: int,
):
    statement = (
        select(Applicant)
        .where(Applicant.id == applicant_id)
    )

    return db.scalar(statement)


def delete_applicant(
    db: Session,
    applicant_id: int,
):
    applicant = get_applicant_by_id(
        db,
        applicant_id,
    )

    if applicant is None:
        return None

    db.delete(applicant)
    db.commit()

    return applicant