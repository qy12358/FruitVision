"""Surface-quality grading rules for an already analysed mango.

This module deliberately does not detect blemishes or damage.  It receives
the quantified coverage percentages from the blemish detector and applies the
requirements in the grading specification.
"""

import math


class QualityGrader:
    """Assign the requirements-based mango surface-quality grade."""

    PREMIUM_LIMIT = 5.0
    MAXIMUM_ACCEPTABLE_LIMIT = 10.0
    VALID_RIPENESS = {"ripe", "semi_ripe", "unripe", "rotten"}

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
