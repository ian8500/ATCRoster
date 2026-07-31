from datetime import date
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import app as roster


def test_training_models_store_dual_score_and_safety_event():
    with roster.app.app_context():
        roster.db.create_all()
        unit = roster.Unit.query.first()
        if not unit:
            unit = roster.Unit(name="Training Test Unit", code="TTU")
            roster.db.session.add(unit)
            roster.db.session.flush()
        people = []
        for index in range(2):
            person = roster.Staff(
                unit_id=unit.id, username=f"training-test-{index}",
                password_hash="unused", name=f"Training Person {index}",
                staff_no=f"TR-{index}", is_trainee=index == 0,
            )
            roster.db.session.add(person)
            people.append(person)
        roster.db.session.flush()
        level = roster.TrainingLevel(
            unit_id=unit.id, name="Test level"
        )
        roster.db.session.add(level)
        roster.db.session.flush()
        objective = roster.TrainingObjective(
            unit_id=unit.id, level_id=level.id, position=1,
            title="Maintain separation", description="Apply minima.",
        )
        roster.db.session.add(objective)
        roster.db.session.flush()
        report = roster.TrainingSession(
            unit_id=unit.id, trainee_id=people[0].id,
            ojti_id=people[1].id, level_id=level.id,
            training_date=date.today(), duration_minutes=90,
        )
        roster.db.session.add(report)
        roster.db.session.flush()
        roster.db.session.add(roster.TrainingScore(
            unit_id=unit.id, session_id=report.id,
            objective_id=objective.id, attainment=3, assistance=2,
            safety_critical=True, note="Instructor intervened.",
        ))
        roster.db.session.commit()
        saved = roster.TrainingScore.query.filter_by(session_id=report.id).one()
        assert (saved.attainment, saved.assistance, saved.safety_critical) == (3, 2, True)
