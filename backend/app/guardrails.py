import re


MAX_QUESTION_LENGTH = 500


SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?instructions",
    r"reveal\s+your\s+system\s+prompt",
    r"show\s+your\s+hidden\s+instructions",
    r"reveal\s+hidden\s+instructions",
    r"bypass\s+your\s+rules",
    r"disregard\s+your\s+instructions",
]


def validate_question(
    question: str
) -> tuple[bool, str]:
    """
    Validate the user's question.
    """

    if not isinstance(question, str):

        return (
            False,
            "Question must be a string."
        )

    question = question.strip()

    if not question:

        return (
            False,
            "Question cannot be empty."
        )

    if len(question) > MAX_QUESTION_LENGTH:

        return (
            False,
            (
                "Question is too long. "
                f"Maximum length is "
                f"{MAX_QUESTION_LENGTH} characters."
            )
        )

    return True, ""


def detect_prompt_injection(
    question: str
) -> bool:
    """
    Detect common prompt-injection patterns.

    This is a basic heuristic and does not
    guarantee complete prompt-injection detection.
    """

    question_lower = question.lower()

    for pattern in SUSPICIOUS_PATTERNS:

        if re.search(
            pattern,
            question_lower
        ):

            return True

    return False


def apply_input_guardrails(
    question: str
) -> str:
    """
    Validate and normalize a question.
    """

    is_valid, error = validate_question(
        question
    )

    if not is_valid:

        raise ValueError(error)

    if detect_prompt_injection(question):

        raise ValueError(
            "This request cannot be processed."
        )

    return question.strip()