import re
from typing import Dict, List, Tuple
from collections import Counter

from pdf_parser import SKILLS_SET, extract_skills


def get_matched_skills(resume_text: str, job_text: str) -> List[str]:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    return sorted(resume_skills & job_skills)


def get_missing_skills(resume_text: str, job_text: str) -> List[str]:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    return sorted(job_skills - resume_skills)


def get_redundant_skills(resume_text: str, job_text: str) -> List[str]:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    return sorted(resume_skills - job_skills - {"communication", "teamwork", "leadership"})


def get_keyword_overlap(resume_text: str, job_text: str) -> Dict:
    resume_tokens = _tokenize(resume_text)
    job_tokens = _tokenize(job_text)

    resume_set = set(resume_tokens)
    job_set = set(job_tokens)

    common = resume_set & job_set
    total = min(len(resume_set), len(job_set))
    overlap_ratio = round(len(common) / total, 4) if total > 0 else 0.0

    return {
        "overlap_ratio": overlap_ratio,
        "common_keywords": sorted(common)[:30],
        "resume_token_count": len(resume_set),
        "job_token_count": len(job_set),
    }


def generate_explanation(
    matched: List[str],
    missing: List[str],
    redundant: List[str],
    fit_prob: float,
    threshold_info: Dict,
) -> str:
    lines = []

    if threshold_info["label"] == "Strong Fit":
        lines.append("Strong alignment between your resume and the job description.")
    elif threshold_info["label"] == "Potential Fit":
        lines.append("Partial match. Your resume covers some key requirements but has gaps.")
    else:
        lines.append("Your resume does not closely match this job description.")

    if matched:
        skills_str = ", ".join(matched[:10])
        lines.append(f"Matched skills ({len(matched)}): {skills_str}.")
    else:
        lines.append("No specific skill matches found based on keyword analysis.")

    if missing:
        skills_str = ", ".join(missing[:8])
        lines.append(f"Missing skills ({len(missing)}): {skills_str}.")
    else:
        lines.append("All required skills appear to be covered.")

    if redundant:
        skills_str = ", ".join(redundant[:5])
        lines.append(f"Consider emphasizing skills more relevant to this role: {skills_str}.")

    return "\n\n".join(lines)


def generate_suggestions(
    missing: List[str],
    redundant: List[str],
    fit_prob: float,
) -> List[str]:
    suggestions = []

    if missing:
        suggestions.append(f"Add experience with: {', '.join(missing[:6])}.")
        suggestions.append("Highlight transferable skills that relate to these missing areas.")

    if redundant:
        suggestions.append(
            f"Reorganize your resume to prioritize skills listed in the job description "
            f"over: {', '.join(redundant[:3])}."
        )

    if fit_prob < 0.4:
        suggestions.append(
            "Consider tailoring your resume summary to match the specific role title "
            "and key requirements."
        )

    if 0.4 <= fit_prob < 0.7:
        suggestions.append(
            "Add specific metrics or project outcomes that demonstrate proficiency in "
            "the matched skills."
        )

    if not suggestions:
        suggestions.append("Your resume is well-aligned. Highlight measurable achievements.")

    return suggestions


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "she", "it", "they", "them",
        "and", "or", "but", "if", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "between", "into",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "in", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "just", "also", "now",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]


if __name__ == "__main__":
    resume = "Python developer skilled in Django, AWS, PostgreSQL, and Docker."
    job = "Looking for a backend engineer with Python, Django, Kubernetes, and Redis."

    matched = get_matched_skills(resume, job)
    missing = get_missing_skills(resume, job)
    redundant = get_redundant_skills(resume, job)
    overlap = get_keyword_overlap(resume, job)
    explanation = generate_explanation(matched, missing, redundant, 0.72, {"label": "Strong Fit"})
    suggestions = generate_suggestions(missing, redundant, 0.72)

    print(f"Matched: {matched}")
    print(f"Missing: {missing}")
    print(f"Redundant: {redundant}")
    print(f"Overlap: {overlap}")
    print(f"\nExplanation:\n{explanation}")
    print(f"\nSuggestions:")
    for s in suggestions:
        print(f"  - {s}")
