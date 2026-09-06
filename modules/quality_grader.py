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

    # This is the grading specification supplied for the application.  Keep
    # the interpretations alongside the thresholds so the result is
    # auditable in the UI and in saved assessment records.
    RULES = (
        {
            "ripeness": "ripe",
            "blemish": "<=5%",
            "damage": "<=5%",
            "grade": "Premium",
            "interpretation": "Best maturity condition and minimal surface defects",
        },
        {
            "ripeness": "semi_ripe",
            "blemish": "<=5%",
            "damage": "<=5%",
            "grade": "Grade 1",
            "interpretation": "Good quality but not yet fully ripe",
        },
        {
            "ripeness": "unripe",
            "blemish": "<=5%",
            "damage": "<=5%",
            "grade": "Grade 2",
            "interpretation": "Clean surface but immature",
        },
        {
            "ripeness": "ripe",
            "blemish": ">5% and <=10%",
            "damage": "<=5%",
            "grade": "Grade 1",
            "interpretation": "Good maturity with moderate cosmetic blemish",
        },
        {
            "ripeness": "semi_ripe",
            "blemish": ">5% and <=10%",
            "damage": "<=5%",
            "grade": "Grade 2",
            "interpretation": "Incomplete maturity plus noticeable blemish",
        },
        {
            "ripeness": "unripe",
            "blemish": ">5% and <=10%",
            "damage": "<=5%",
            "grade": "Grade 2",
            "interpretation": "Already limited by immature condition",
        },
        {
            "ripeness": "ripe",
            "blemish": "<=10%",
            "damage": ">5% and <=10%",
            "grade": "Grade 2",
            "interpretation": "Correct maturity but noticeable physical damage",
        },
        {
            "ripeness": "semi_ripe",
            "blemish": "<=10%",
            "damage": ">5% and <=10%",
            "grade": "Grade 2",
            "interpretation": "Incomplete maturity and damage",
        },
        {
            "ripeness": "unripe",
            "blemish": "<=10%",
            "damage": ">5% and <=10%",
            "grade": "Grade 2",
            "interpretation": "Immature and damaged",
        },
        {
            "ripeness": "any",
            "blemish": ">10%",
            "damage": "Any",
            "grade": "Reject",
            "interpretation": "Excessive blemish",
        },
        {
            "ripeness": "any",
            "blemish": "Any",
            "damage": ">10%",
            "grade": "Reject",
            "interpretation": "Excessive damage",
        },
        {
            "ripeness": "rotten",
            "blemish": "Any",
            "damage": "Any",
            "grade": "Reject",
            "interpretation": "Deteriorated fruit",
        },
    )

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
            matched_rule = self.RULES[-1]
        elif blemish > self.MAXIMUM_ACCEPTABLE_LIMIT:
            matched_rule = self.RULES[-3]
        elif damage > self.MAXIMUM_ACCEPTABLE_LIMIT:
            matched_rule = self.RULES[-2]
        elif blemish <= self.PREMIUM_LIMIT and damage <= self.PREMIUM_LIMIT:
            matched_rule = {
                "ripe": self.RULES[0],
                "semi_ripe": self.RULES[1],
                "unripe": self.RULES[2],
            }[ripeness_text]
        elif blemish > self.PREMIUM_LIMIT and damage <= self.PREMIUM_LIMIT:
            matched_rule = {
                "ripe": self.RULES[3],
                "semi_ripe": self.RULES[4],
                "unripe": self.RULES[5],
            }[ripeness_text]
        else:
            matched_rule = {
                "ripe": self.RULES[6],
                "semi_ripe": self.RULES[7],
                "unripe": self.RULES[8],
            }[ripeness_text]

        grade = matched_rule["grade"]
        interpretation = matched_rule["interpretation"]
        reasons = [interpretation]

        return {
            "available": True,
            "grade": grade,
            "blemish_coverage": round(blemish, 2),
            "damage_coverage": round(damage, 2),
            "ripeness": ripeness,
            "reason": " ".join(reasons),
            "reasons": reasons,
            "interpretation": interpretation,
            "rule": dict(matched_rule),
        }


def grade_surface_quality(blemish_coverage, damage_coverage, ripeness):
    """Functional wrapper used by the application and tests."""

    return QualityGrader().grade(
        blemish_coverage=blemish_coverage,
        damage_coverage=damage_coverage,
        ripeness=ripeness,
    )
