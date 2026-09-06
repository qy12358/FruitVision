"""Surface-quality grading rules for an already analysed mango.

This module deliberately does not detect blemishes or damage.  It receives
the quantified coverage percentages from the blemish detector and applies the
requirements in the grading specification.
"""

import math

LEGACY_SURFACE_GRADING_RULES = [
    ('Less than 1.50%', 'Low', 'Premium'),
    ('1.50% to less than 5.00%', 'Medium', 'Grade 1'),
    ('5.00% to 10.00% inclusive', 'High', 'Grade 2'),
    ('Above 10.00%', 'High', 'Reject'),
]

SURFACE_GRADING_HEADERS = ('Ripeness stage', 'Total defect <= 5%',
                           'Total defect > 5% to <= 10%', 'Total defect > 10%')
SURFACE_GRADING_RULES = [
    ('Ripe', 'Premium', 'Grade 1', 'Reject'),
    ('Semi-Ripe', 'Grade 1', 'Grade 2', 'Reject'),
    ('Unripe', 'Grade 2', 'Grade 2', 'Reject'),
    ('Rotten', 'Reject', 'Reject', 'Reject'),
]


def surface_severity(coverage):
    return 'Low' if coverage < 1.5 else 'Medium' if coverage < 5.0 else 'High'


class QualityGrader:
    """Assign the requirements-based mango surface-quality grade."""

    PREMIUM_LIMIT = 5.0
    MAXIMUM_ACCEPTABLE_LIMIT = 10.0
    VALID_RIPENESS = {"ripe", "semi_ripe", "unripe", "rotten"}

    def grade_defects(self, defect_coverage, ripeness=None):
        """Apply the ripeness/total-defect matrix; severity remains coverage-only."""
        errors = []
        coverage = self._validate_percentage('Defect coverage', defect_coverage, errors)
        stage = self._normalise_ripeness(ripeness)
        if stage not in self.VALID_RIPENESS:
            errors.append('Ripeness must be Ripe, Semi-Ripe, Unripe, or Rotten')
        if errors:
            return dict(available=False, grade='Unavailable', defect_coverage=coverage,
                        ripeness=ripeness,
                        severity=surface_severity(coverage) if coverage is not None else 'Unavailable',
                        reason='; '.join(errors), reasons=errors,
                        grading_basis='surface_defects_v2')
        severity = surface_severity(coverage)
        row = next(row for row in SURFACE_GRADING_RULES
                   if self._normalise_ripeness(row[0]) == stage)
        column = 1 if coverage <= 5 else 2 if coverage <= 10 else 3
        grade = row[column]
        reason = (f'Detected surface defects cover {coverage:.2f}% of the visible mango area. '
                  f'{row[0]} ripeness with {SURFACE_GRADING_HEADERS[column].lower()} gives {grade}.')
        if stage == 'rotten':
            reason = f'Rotten mangoes are rejected regardless of total defect coverage ({coverage:.2f}%).'
        return dict(available=True, grade=grade, defect_coverage=coverage,
                    blemish_coverage=coverage, damage_coverage=None, severity=severity,
                    ripeness=ripeness, reason=reason, reasons=[reason], grading_basis='surface_defects_v2')

    @staticmethod
    def _normalise_ripeness(value):
        if value is None:
            return ""
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _validate_percentage(name, value, errors):
        if value is None or isinstance(value, bool):
            errors.append(f"{name} is unavailable")
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} is not numeric")
            return None
        if not math.isfinite(number) or not 0.0 <= number <= 100.0:
            errors.append(f"{name} must be between 0% and 100%")
            return None
        return number

    def grade(
        self,
        blemish_coverage,
        damage_coverage,
        ripeness,
    ):
        """Return an auditable grading result.

        The result remains unavailable when an input is missing or invalid;
        invalid data is never silently treated as zero damage.
        """

        errors = []
        blemish = self._validate_percentage(
            "Blemish coverage", blemish_coverage, errors
        )
        damage = self._validate_percentage(
            "Damage coverage", damage_coverage, errors
        )
        ripeness_text = self._normalise_ripeness(ripeness)
        if not ripeness_text:
            errors.append("Ripeness classification is unavailable")
        elif ripeness_text not in self.VALID_RIPENESS:
            errors.append(
                "Ripeness must be Ripe, Semi-Ripe, Unripe, or Rotten"
            )

        if errors:
            return {
                "available": False,
                "grade": "Unavailable",
                "blemish_coverage": blemish,
                "damage_coverage": damage,
                "ripeness": ripeness,
                "reason": "; ".join(errors),
                "reasons": errors,
            }

        if ripeness_text == "rotten":
            grade = "Reject"
            reasons = ["Deteriorated fruit is Rotten and must be rejected."]
        elif blemish > self.MAXIMUM_ACCEPTABLE_LIMIT:
            grade = "Reject"
            reasons = [
                "Excessive blemish: coverage is above 10%.",
            ]
        elif damage > self.MAXIMUM_ACCEPTABLE_LIMIT:
            grade = "Reject"
            reasons = [
                "Excessive damage: coverage is above 10%.",
            ]
        else:
            low_blemish = blemish <= self.PREMIUM_LIMIT
            low_damage = damage <= self.PREMIUM_LIMIT

            if low_blemish and low_damage:
                grade, reasons = {
                    "ripe": (
                        "Premium",
                        [
                            "Optimal eating stage with minimal visible "
                            "defect/damage."
                        ],
                    ),
                    "semi_ripe": (
                        "Grade 1",
                        ["Good condition but not yet at optimal ripe stage."],
                    ),
                    "unripe": (
                        "Grade 2",
                        [
                            "Surface may be excellent, but fruit is not "
                            "ready for normal consumption."
                        ],
                    ),
                }[ripeness_text]
            elif low_damage and blemish <= self.MAXIMUM_ACCEPTABLE_LIMIT:
                grade, reasons = {
                    "ripe": (
                        "Grade 1",
                        [
                            "Good ripeness but noticeable cosmetic defects."
                        ],
                    ),
                    "semi_ripe": (
                        "Grade 2",
                        [
                            "Both maturity and appearance reduce overall "
                            "quality."
                        ],
                    ),
                    "unripe": (
                        "Grade 2",
                        ["Already limited by immature condition."],
                    ),
                }[ripeness_text]
            elif damage <= self.MAXIMUM_ACCEPTABLE_LIMIT:
                grade, reasons = {
                    "ripe": (
                        "Grade 2",
                        ["Appropriate ripeness but significant damage."],
                    ),
                    "semi_ripe": (
                        "Grade 2",
                        ["Damage and incomplete ripeness reduce quality."],
                    ),
                    "unripe": (
                        "Grade 2",
                        ["Lowest acceptable overall quality."],
                    ),
                }[ripeness_text]

        return {
            "available": True,
            "grade": grade,
            "blemish_coverage": round(blemish, 2),
            "damage_coverage": round(damage, 2),
            "ripeness": ripeness,
            "reason": " ".join(reasons),
            "reasons": reasons,
        }


def grade_surface_quality(blemish_coverage, damage_coverage, ripeness):
    """Functional wrapper used by the application and tests."""

    return QualityGrader().grade(
        blemish_coverage=blemish_coverage,
        damage_coverage=damage_coverage,
        ripeness=ripeness,
    )
