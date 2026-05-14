import re
import logging
from typing import List, Dict, Optional, Set

from preprocessing import clean_text

logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:
    fitz = None
    logger.warning("PyMuPDF (fitz) not installed. PDF parsing will raise errors.")

# ---------------------------------------------------------------------------
# Skill lexicon
# ---------------------------------------------------------------------------
SKILLS_SET: Set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby",
    "go", "rust", "swift", "kotlin", "scala", "php", "html", "css",
    "sql", "nosql", "bash", "powershell", "shell", "perl", "r",
    "matlab", "julia", "dart", "lua",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "elasticsearch",
    "dynamodb", "mariadb", "sqlite", "oracle", "sql server",
    # Frontend
    "react", "angular", "vue", "svelte", "next.js", "nuxt", "gatsby",
    "redux", "jquery", "bootstrap", "tailwind", "sass", "less",
    "webpack", "vite", "babel", "storybook",
    # Backend / Frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "node.js",
    "express", "asp.net", "laravel", "ruby on rails", "gin",
    "echo", "fiber", "actix", "rocket",
    # ML / Data Science
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "scipy", "matplotlib", "seaborn", "plotly", "dash", "streamlit",
    "tableau", "power bi", "jupyter", "spark", "hadoop", "airflow",
    "mlflow", "kubeflow", "onnx", "opencv", "hugging face",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "data analysis", "data engineering",
    "artificial intelligence", "generative ai", "llm", "rag",
    "langchain", "llamaindex", "vector database", "chroma", "pinecone",
    # Cloud / DevOps
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
    "pulumi", "ansible", "jenkins", "github actions", "gitlab ci",
    "circleci", "travis ci", "argo", "helm", "istio",
    "git", "github", "gitlab", "bitbucket",
    "ci/cd", "microservices", "serverless", "lambda",
    # Tools / Methodologies
    "jira", "confluence", "notion", "slack",
    "agile", "scrum", "kanban", "waterfall",
    "rest api", "graphql", "grpc", "soap", "kafka", "rabbitmq",
    "nginx", "apache", "linux", "unix", "vim", "vscode",
    # Office / Design
    "excel", "word", "powerpoint", "outlook", "sharepoint",
    "photoshop", "figma", "sketch", "adobe xd", "illustrator",
    "ui/ux", "product management", "a/b testing",
    # Soft skills
    "leadership", "communication", "teamwork", "problem solving",
    "critical thinking", "project management", "time management",
    "mentoring", "presentation", "negotiation",
}

# ---------------------------------------------------------------------------
# Education lexicon
# ---------------------------------------------------------------------------
EDUCATION_KEYWORDS: Set[str] = {
    "bachelor", "master", "phd", "doctorate", "associate",
    "b.s.", "b.a.", "b.eng", "b.sc", "b.tech",
    "m.s.", "m.a.", "m.eng", "m.sc", "m.tech", "mba",
    "ph.d", "ph.d.",
    "high school", "diploma", "graduate", "postgraduate",
    "bachelor of science", "bachelor of arts", "bachelor of engineering",
    "bachelor of technology", "bachelor of business administration",
    "master of science", "master of arts", "master of engineering",
    "master of technology", "master of business administration",
    "doctor of philosophy",
}

DEGREE_KEYWORDS: Set[str] = {
    "computer science", "computer engineering", "software engineering",
    "information technology", "information systems", "data science",
    "electrical engineering", "mechanical engineering", "civil engineering",
    "chemical engineering", "biomedical engineering",
    "mathematics", "statistics", "physics", "chemistry", "biology",
    "business administration", "finance", "accounting", "economics",
    "marketing", "management", "communications", "psychology",
    "english", "history", "political science", "sociology",
    "philosophy", "economics", "architecture", "design",
}

# ---------------------------------------------------------------------------
# Experience / date patterns
# ---------------------------------------------------------------------------
MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

EXPERIENCE_DATE_PATTERN = re.compile(
    rf"(?:{MONTHS})\s*\d{{4}}\s*(?:-|–|to)\s*"
    rf"(?:(?:{MONTHS})\s*\d{{4}}|present|current)",
    re.IGNORECASE,
)

NUMERIC_DATE_PATTERN = re.compile(
    r"\d{4}\s*(?:-|–|to)\s*(?:\d{4}|present|current)",
    re.IGNORECASE,
)

JOB_TITLE_KEYWORDS: Set[str] = {
    "engineer", "developer", "scientist", "analyst", "manager",
    "director", "lead", "head", "architect", "consultant",
    "specialist", "coordinator", "administrator", "associate",
    "intern", "fellow", "researcher", "instructor", "professor",
    "officer", "executive", "president", "vp", "vice president",
    "technician", "designer", "writer", "editor",
    "software engineer", "data scientist", "product manager",
    "project manager", "team lead", "tech lead",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_resume_text(pdf_path: str) -> str:
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required. Install it with: pip install PyMuPDF"
        )
    doc = fitz.open(pdf_path)
    text_parts: List[str] = []
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text_parts.append(page_text)
    doc.close()
    return "\n".join(text_parts)


def clean_resume_text(text: str) -> str:
    return clean_text(text, lowercase=True, remove_stopwords_flag=False)


def extract_skills(text: str, skills_set: Optional[Set[str]] = None) -> List[str]:
    skills = skills_set if skills_set is not None else SKILLS_SET
    text_lower = text.lower()
    found: Set[str] = set()
    for skill in sorted(skills, key=len, reverse=True):
        if skill in text_lower:
            found.add(skill)
    return sorted(found)


def extract_education(text: str) -> List[Dict[str, Optional[str]]]:
    text_lower = text.lower()
    sentences = [
        s.strip()
        for s in text_lower.replace("\n", " ").split(".")
        if s.strip()
    ]
    education_entries: List[Dict[str, Optional[str]]] = []
    for sentence in sentences:
        degree_found = None
        major_found = None
        for kw in EDUCATION_KEYWORDS:
            if kw in sentence:
                degree_found = kw
                break
        for kw in DEGREE_KEYWORDS:
            if kw in sentence:
                major_found = kw
                break
        if degree_found or major_found:
            education_entries.append(
                {
                    "text": sentence.strip(),
                    "degree": degree_found,
                    "major": major_found,
                }
            )
    return education_entries


def extract_experience(text: str) -> List[Dict[str, str]]:
    text_lower = text.lower()
    lines = text_lower.replace("\r\n", "\n").split("\n")
    experience_entries: List[Dict[str, str]] = []
    current_entry: Dict[str, str] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        date_match = EXPERIENCE_DATE_PATTERN.search(line) or NUMERIC_DATE_PATTERN.search(
            line
        )
        if date_match:
            if current_entry and ("title" in current_entry or "dates" in current_entry):
                experience_entries.append(current_entry)
            current_entry = {"dates": date_match.group()}
            continue

        title_match = None
        for kw in JOB_TITLE_KEYWORDS:
            if kw in line:
                title_match = line
                break

        if title_match:
            if "title" not in current_entry:
                current_entry["title"] = title_match

    if current_entry and ("title" in current_entry or "dates" in current_entry):
        experience_entries.append(current_entry)

    return experience_entries


def parse_resume(pdf_path: str) -> Dict:
    raw_text = extract_resume_text(pdf_path)
    cleaned_text = clean_resume_text(raw_text)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "skills": extract_skills(cleaned_text),
        "education": extract_education(cleaned_text),
        "experience": extract_experience(cleaned_text),
    }
