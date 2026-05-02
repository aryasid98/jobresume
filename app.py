"""
Resume ↔ Job Matcher (LLM via Ollama/Groq/OpenAI, ChromaDB Vector Store)

Universal skill extraction for ALL industries:
- Technical (Software, IT, Engineering)
- Business (MBA, Management, Consulting)
- Finance (Banking, Accounting, Investment)
- Healthcare (Medical, Nursing, Pharma)
- Marketing & Sales
- Legal, HR, Education, and more
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
load_dotenv()
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import gspread
import gc
import httpx
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "Resume ↔ Job Matcher (All Industries)"
DB_PATH = os.environ.get("JOBRESUME_DB_PATH", "jobresume.db")

# LLM configuration
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # override default model per provider
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# Ollama-specific (used when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# ChromaDB configuration
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
ENABLE_CHROMA = os.environ.get("ENABLE_CHROMA", "true").lower() == "true"

# Google Drive configuration
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
USE_GOOGLE_DRIVE = os.environ.get("USE_GOOGLE_DRIVE", "false").lower() == "true"

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_BUCKET_NAME = os.environ.get("SUPABASE_BUCKET_NAME", "resumes")
USE_SUPABASE = os.environ.get("USE_SUPABASE", "false").lower() == "true"
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")
USE_SUPABASE_DB = os.environ.get("USE_SUPABASE_DB", "false").lower() == "true"

# Conditionally import ChromaDB to save memory when disabled
if ENABLE_CHROMA:
    import chromadb
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from google.oauth2.service_account import Credentials
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Add remaining configuration that wasn't in the first section
EMBEDDING_MODEL_NAME = os.environ.get("JOBRESUME_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TEXT_CHARS = int(os.environ.get("JOBRESUME_MAX_TEXT_CHARS", "20000"))

# Google Sheets configuration
GOOGLE_SHEETS_CREDENTIALS_FILE = os.environ.get("GOOGLE_SHEETS_CREDENTIALS", "google_credentials.json")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "1ghqj2QQytGrUh97PRhaNmYFxG8z7JybUA91GHWIfUys")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Sheet1")
JOB_RETENTION_DAYS = int(os.environ.get("JOB_RETENTION_DAYS", "30"))


# =============================================================================
# Prompts - Universal for ALL Industries
# =============================================================================

RESUME_SKILL_EXTRACTION_PROMPT = """You are an expert resume parser that works across ALL industries and job types.

Analyze this resume and extract ALL skills explicitly mentioned - technical, business, soft skills, domain expertise, certifications, tools, and competencies.

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

Extract ONLY skills explicitly mentioned in the resume. Return empty arrays [] if a category has no matching skills.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "name": null,
  "email": null,
  "phone": null,
  "technical_skills": [],
  "business_skills": [],
  "domain_expertise": [],
  "soft_skills": [],
  "tools_and_software": [],
  "certifications": [],
  "languages": [],
  "experience_years": null,
  "education": [],
  "job_titles": [],
  "industries": [],
  "summary": ""
}}

RULES:
1. Only extract skills EXPLICITLY stated in the resume
2. Do NOT invent or assume skills that are not mentioned
3. Use empty arrays [] for categories with no matching skills
4. This could be ANY profession - engineer, MBA, doctor, lawyer, teacher, sales, etc.

Return ONLY the JSON object, nothing else."""


JOB_SKILL_EXTRACTION_PROMPT = """You are an expert job description parser that works across ALL industries.

Analyze this job posting and extract ALL required and preferred qualifications - technical skills, business skills, soft skills, certifications, experience requirements.

JOB TITLE: {job_title}

JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"

Extract ONLY skills explicitly mentioned in the job description. Return empty arrays [] if a category is not mentioned.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "required_technical_skills": [],
  "required_business_skills": [],
  "required_domain_expertise": [],
  "required_soft_skills": [],
  "preferred_technical_skills": [],
  "preferred_business_skills": [],
  "preferred_certifications": [],
  "required_certifications": [],
  "required_education": [],
  "experience_years_min": null,
  "experience_years_max": null,
  "industry": null,
  "job_type": null,
  "seniority_level": null,
  "department": null,
  "summary": ""
}}

RULES:
1. Only extract skills EXPLICITLY stated in the job description
2. Do NOT invent or assume skills that are not mentioned
3. Use empty arrays [] for categories with no matching skills
4. This job could be in ANY field - software, finance, healthcare, legal, marketing, HR, etc.

Return ONLY the JSON object, nothing else."""


MATCH_EXPLANATION_PROMPT = """You are a career advisor. Explain why this candidate matches or doesn't match this job.

CANDIDATE PROFILE:
- Technical Skills: {candidate_technical}
- Business Skills: {candidate_business}
- Domain Expertise: {candidate_domain}
- Certifications: {candidate_certs}
- Experience: {candidate_experience} years
- Industries: {candidate_industries}

JOB REQUIREMENTS:
- Required Technical: {job_technical}
- Required Business: {job_business}
- Required Domain: {job_domain}
- Required Certifications: {job_certs}
- Experience: {job_experience_min}-{job_experience_max} years
- Industry: {job_industry}

Provide a 2-3 sentence match analysis. Be specific about overlapping skills and gaps."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResumeData:
    """Structured data extracted from a resume - supports all industries."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    technical_skills: List[str] = field(default_factory=list)
    business_skills: List[str] = field(default_factory=list)
    domain_expertise: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)
    tools_and_software: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    experience_years: Optional[int] = None
    education: List[str] = field(default_factory=list)
    job_titles: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data: str) -> "ResumeData":
        try:
            d = json.loads(data)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def all_skills(self) -> List[str]:
        """Get all skills combined (excludes soft skills for matching)."""
        return (
            self.technical_skills +
            self.business_skills +
            self.domain_expertise +
            self.tools_and_software +
            self.certifications
        )

    def skills_for_embedding(self) -> str:
        """Create text representation for embedding (excludes soft skills)."""
        parts = []
        if self.technical_skills:
            parts.append("Technical: " + ", ".join(self.technical_skills))
        if self.business_skills:
            parts.append("Business: " + ", ".join(self.business_skills))
        if self.domain_expertise:
            parts.append("Domain: " + ", ".join(self.domain_expertise))
        if self.tools_and_software:
            parts.append("Tools: " + ", ".join(self.tools_and_software))
        if self.certifications:
            parts.append("Certifications: " + ", ".join(self.certifications))
        if self.job_titles:
            parts.append("Roles: " + ", ".join(self.job_titles))
        if self.industries:
            parts.append("Industries: " + ", ".join(self.industries))
        return "\n".join(parts)


@dataclass
class JobData:
    """Structured data extracted from a job description - supports all industries."""
    required_technical_skills: List[str] = field(default_factory=list)
    required_business_skills: List[str] = field(default_factory=list)
    required_domain_expertise: List[str] = field(default_factory=list)
    required_soft_skills: List[str] = field(default_factory=list)
    preferred_technical_skills: List[str] = field(default_factory=list)
    preferred_business_skills: List[str] = field(default_factory=list)
    preferred_certifications: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)
    required_education: List[str] = field(default_factory=list)
    experience_years_min: Optional[int] = None
    experience_years_max: Optional[int] = None
    industry: Optional[str] = None
    job_type: Optional[str] = None
    seniority_level: Optional[str] = None
    department: Optional[str] = None
    summary: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data: str) -> "JobData":
        try:
            d = json.loads(data)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def all_required_skills(self) -> List[str]:
        """Get all required skills (excludes soft skills for matching)."""
        return (
            self.required_technical_skills +
            self.required_business_skills +
            self.required_domain_expertise +
            self.required_certifications
        )

    def all_preferred_skills(self) -> List[str]:
        return (
            self.preferred_technical_skills +
            self.preferred_business_skills +
            self.preferred_certifications
        )

    def skills_for_embedding(self) -> str:
        """Create text representation for embedding (excludes soft skills)."""
        parts = []
        if self.required_technical_skills:
            parts.append("Required Technical: " + ", ".join(self.required_technical_skills))
        if self.required_business_skills:
            parts.append("Required Business: " + ", ".join(self.required_business_skills))
        if self.required_domain_expertise:
            parts.append("Required Domain: " + ", ".join(self.required_domain_expertise))
        if self.required_certifications:
            parts.append("Required Certs: " + ", ".join(self.required_certifications))
        if self.preferred_technical_skills:
            parts.append("Preferred Technical: " + ", ".join(self.preferred_technical_skills))
        if self.preferred_business_skills:
            parts.append("Preferred Business: " + ", ".join(self.preferred_business_skills))
        if self.industry:
            parts.append("Industry: " + self.industry)
        if self.department:
            parts.append("Department: " + self.department)
        return "\n".join(parts)


# =============================================================================
# LLM Clients (Ollama, Groq, OpenAI)
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Send a prompt. Returns (response_text, error_message)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""

    @abstractmethod
    def get_provider_name(self) -> str:
        """Human-readable provider name for status display."""

    @staticmethod
    def extract_json_from_response(text: str) -> Optional[dict]:
        """Extract JSON object from LLM response."""
        if not text:
            return None
        # Try code blocks first
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            text = code_block_match.group(1)
        # Find raw JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None


class OllamaClient(LLMClient):
    """Client for interacting with local Ollama LLM."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            }
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", ""), None
        except httpx.ConnectError:
            return None, f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
        except httpx.TimeoutException:
            return None, f"Ollama request timed out after {self.timeout}s"
        except Exception as e:
            return None, f"Ollama error: {str(e)}"

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    return self.model.split(":")[0] in model_names
        except Exception:
            pass
        return False

    def get_provider_name(self) -> str:
        return f"Ollama ({self.model})"


class GroqClient(LLMClient):
    """Client for Groq cloud API (OpenAI-compatible)."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str = LLM_API_KEY, model: str = "", timeout: int = LLM_TIMEOUT):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.base_url = "https://api.groq.com/openai/v1"

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.api_key:
            return None, "Groq API key not set. Set LLM_API_KEY environment variable."

        import time
        max_retries = 5
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 4096,
                        }
                    )
                    if resp.status_code == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Groq rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            return None, f"Groq API rate limit exceeded after {max_retries} retries"
                    elif resp.status_code == 404:
                        return None, f"Groq model '{self.model}' not found. Check LLM_MODEL environment variable."
                    elif resp.status_code == 400:
                        error_detail = resp.text if resp.text else "No error details"
                        return None, f"Groq API bad request: {error_detail}"
                    resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"], None
            except httpx.ConnectError:
                return None, "Cannot connect to Groq API."
            except httpx.TimeoutException:
                return None, f"Groq API request timed out after {self.timeout}s"
            except Exception as e:
                return None, f"Groq API error: {str(e)}"

        return None, f"Groq API failed after {max_retries} retries"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                print(f"Groq API status: {resp.status_code}, response: {resp.text[:200]}")
                return resp.status_code == 200
        except Exception as e:
            print(f"Groq API error: {e}")
            return False

    def get_provider_name(self) -> str:
        return f"Groq ({self.model})"


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str = LLM_API_KEY, model: str = "", timeout: int = LLM_TIMEOUT):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.base_url = "https://api.openai.com/v1"

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.api_key:
            return None, "OpenAI API key not set. Set LLM_API_KEY environment variable."
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"], None
        except httpx.ConnectError:
            return None, "Cannot connect to OpenAI API."
        except httpx.TimeoutException:
            return None, f"OpenAI API request timed out after {self.timeout}s"
        except Exception as e:
            return None, f"OpenAI API error: {str(e)}"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return resp.status_code == 200
        except Exception:
            return False

    def get_provider_name(self) -> str:
        return f"OpenAI ({self.model})"


def create_llm_client() -> LLMClient:
    """Factory: create the appropriate LLM client based on LLM_PROVIDER."""
    provider = LLM_PROVIDER.lower().strip()
    if provider == "groq":
        return GroqClient(api_key=LLM_API_KEY, model=LLM_MODEL or "", timeout=LLM_TIMEOUT)
    elif provider == "openai":
        return OpenAIClient(api_key=LLM_API_KEY, model=LLM_MODEL or "", timeout=LLM_TIMEOUT)
    else:
        return OllamaClient()


# =============================================================================
# Skill Extractor (uses LLM)
# =============================================================================

class SkillExtractor:
    """Extracts skills from resumes and jobs using LLM - works for ALL industries."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_from_resume(self, resume_text: str) -> Tuple[ResumeData, Optional[str]]:
        """Extract structured data from resume. Returns (ResumeData, error)."""
        if not resume_text or not resume_text.strip():
            return ResumeData(), "Empty resume text"

        prompt = RESUME_SKILL_EXTRACTION_PROMPT.format(
            resume_text=resume_text[:MAX_TEXT_CHARS]
        )
        response, error = self.llm.generate(prompt)
        if error:
            return ResumeData(), error

        parsed = self.llm.extract_json_from_response(response)
        if not parsed:
            return ResumeData(), "Failed to parse LLM response as JSON"

        return ResumeData(
            name=parsed.get("name"),
            email=parsed.get("email"),
            phone=parsed.get("phone"),
            technical_skills=parsed.get("technical_skills", []) or [],
            business_skills=parsed.get("business_skills", []) or [],
            domain_expertise=parsed.get("domain_expertise", []) or [],
            soft_skills=parsed.get("soft_skills", []) or [],
            tools_and_software=parsed.get("tools_and_software", []) or [],
            certifications=parsed.get("certifications", []) or [],
            languages=parsed.get("languages", []) or [],
            experience_years=parsed.get("experience_years"),
            education=parsed.get("education", []) or [],
            job_titles=parsed.get("job_titles", []) or [],
            industries=parsed.get("industries", []) or [],
            summary=parsed.get("summary"),
        ), None

    def extract_from_job(self, job_title: str, job_description: str) -> Tuple[JobData, Optional[str]]:
        """Extract structured data from job description. Returns (JobData, error)."""
        if not job_description or not job_description.strip():
            return JobData(), "Empty job description"

        prompt = JOB_SKILL_EXTRACTION_PROMPT.format(
            job_title=job_title,
            job_description=job_description[:MAX_TEXT_CHARS]
        )
        response, error = self.llm.generate(prompt)
        if error:
            return JobData(), error

        parsed = self.llm.extract_json_from_response(response)
        if not parsed:
            return JobData(), "Failed to parse LLM response as JSON"

        return JobData(
            required_technical_skills=parsed.get("required_technical_skills", []) or [],
            required_business_skills=parsed.get("required_business_skills", []) or [],
            required_domain_expertise=parsed.get("required_domain_expertise", []) or [],
            required_soft_skills=parsed.get("required_soft_skills", []) or [],
            preferred_technical_skills=parsed.get("preferred_technical_skills", []) or [],
            preferred_business_skills=parsed.get("preferred_business_skills", []) or [],
            preferred_certifications=parsed.get("preferred_certifications", []) or [],
            required_certifications=parsed.get("required_certifications", []) or [],
            required_education=parsed.get("required_education", []) or [],
            experience_years_min=parsed.get("experience_years_min"),
            experience_years_max=parsed.get("experience_years_max"),
            industry=parsed.get("industry"),
            job_type=parsed.get("job_type"),
            seniority_level=parsed.get("seniority_level"),
            department=parsed.get("department"),
            summary=parsed.get("summary"),
        ), None


# =============================================================================
# Document Parser
# =============================================================================

class DocumentParser:
    """Parses PDF and DOCX files to extract text."""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        reader = PdfReader(file_path)
        parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
        return DocumentParser._safe_text("\n".join(parts).strip())

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        doc = DocxDocument(file_path)
        parts = [p.text for p in doc.paragraphs if p.text]
        return DocumentParser._safe_text("\n".join(parts).strip())

    @staticmethod
    def extract_text(file_path: str, original_filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract text from file. Returns (text, error)."""
        name = (original_filename or "").lower()
        try:
            if name.endswith(".pdf"):
                return DocumentParser.extract_text_from_pdf(file_path), None
            if name.endswith(".docx"):
                return DocumentParser.extract_text_from_docx(file_path), None
            if name.endswith(".doc"):
                return None, "Unsupported .doc format. Please upload a .docx or .pdf file."
            return None, "Unsupported file type. Please upload a .pdf or .docx file."
        except Exception as e:
            return None, f"Error parsing file: {str(e)}"

    @staticmethod
    def _safe_text(text: str) -> str:
        text = (text or "").replace("\x00", " ")
        if len(text) > MAX_TEXT_CHARS:
            return text[:MAX_TEXT_CHARS]
        return text


# =============================================================================
# Smart Skill Matching
# =============================================================================

# Generic/common words that should NOT be matched alone - they need context
GENERIC_SKILL_MODIFIERS = {
    "operations", "management", "analysis", "development", "engineering",
    "design", "planning", "strategy", "consulting", "administration",
    "coordination", "support", "services", "solutions", "systems",
    "processing", "handling", "execution", "optimization", "integration",
    "automation", "transformation", "governance", "compliance", "reporting",
    "analytics", "intelligence", "architecture", "infrastructure", "security",
    "quality", "assurance", "testing", "training", "leadership", "communication"
}

# Domain prefixes that completely change the meaning of a skill
DOMAIN_PREFIXES = {
    "hr", "human resources", "b2b", "b2c", "it", "sales", "marketing",
    "finance", "financial", "supply chain", "logistics", "customer",
    "product", "project", "program", "data", "business", "digital",
    "cloud", "network", "software", "hardware", "retail", "healthcare",
    "legal", "compliance", "risk", "credit", "investment", "banking",
    "insurance", "manufacturing", "warehouse", "inventory", "procurement",
    "vendor", "client", "account", "revenue", "cost", "budget", "audit",
    "tax", "payroll", "recruitment", "talent", "performance", "learning",
    "content", "social media", "brand", "campaign", "crm", "erp", "sap"
}


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""
    return skill.lower().strip()


def is_compound_skill(skill: str) -> bool:
    """Check if a skill is a compound term (e.g., 'HR Operations')."""
    words = normalize_skill(skill).split()
    if len(words) < 2:
        return False
    # Check if any word is a generic modifier
    return any(w in GENERIC_SKILL_MODIFIERS for w in words)


def get_skill_domain(skill: str) -> Optional[str]:
    """Extract the domain prefix from a compound skill."""
    normalized = normalize_skill(skill)
    words = normalized.split()
    
    # Check single word prefixes
    if words and words[0] in DOMAIN_PREFIXES:
        return words[0]
    
    # Check two-word prefixes (e.g., "supply chain", "human resources")
    if len(words) >= 2:
        two_word = f"{words[0]} {words[1]}"
        if two_word in DOMAIN_PREFIXES:
            return two_word
    
    return None


def skills_match(skill1: str, skill2: str) -> bool:
    """
    Smart skill matching that handles compound terms correctly.
    'HR Operations' should NOT match 'B2B Operations'.
    'Python' should match 'Python'.
    'Project Management' should match 'Project Management'.
    """
    s1 = normalize_skill(skill1)
    s2 = normalize_skill(skill2)
    
    # Exact match
    if s1 == s2:
        return True
    
    # If both are compound skills, they must have the same domain
    if is_compound_skill(skill1) and is_compound_skill(skill2):
        domain1 = get_skill_domain(skill1)
        domain2 = get_skill_domain(skill2)
        
        # If domains are different, skills don't match
        if domain1 and domain2 and domain1 != domain2:
            return False
        
        # If one has a domain and other doesn't, they don't match
        if (domain1 and not domain2) or (domain2 and not domain1):
            return False
    
    # Check if one is a substring of the other (for variations)
    # But only if neither is a generic modifier alone
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    # Don't match if the only overlap is generic modifiers
    overlap = words1 & words2
    if overlap and overlap.issubset(GENERIC_SKILL_MODIFIERS):
        return False
    
    return False


def calculate_skill_overlap(candidate_skills: List[str], job_skills: List[str]) -> Tuple[List[str], List[str], float]:
    """
    Calculate smart skill overlap between candidate and job.
    Returns (matched_skills, missing_skills, match_ratio).
    """
    matched = []
    missing = []
    
    candidate_normalized = {normalize_skill(s): s for s in candidate_skills}
    
    for job_skill in job_skills:
        job_norm = normalize_skill(job_skill)
        found = False
        
        for cand_norm, cand_original in candidate_normalized.items():
            if skills_match(job_skill, cand_original):
                matched.append(job_skill)
                found = True
                break
        
        if not found:
            missing.append(job_skill)
    
    match_ratio = len(matched) / len(job_skills) if job_skills else 0.0
    return matched, missing, match_ratio


# Common abbreviations and synonyms for fuzzy matching
SKILL_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "cpp": "c++",
    "c#": "csharp",
    "csharp": "c#",
    "golang": "go",
    "k8s": "kubernetes",
    "tf": "terraform",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "db": "database",
    "sql server": "microsoft sql server",
    "mssql": "microsoft sql server",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "react": "reactjs",
    "react.js": "reactjs",
    "vue": "vuejs",
    "vue.js": "vuejs",
    "angular": "angularjs",
    "angular.js": "angularjs",
    "node": "nodejs",
    "node.js": "nodejs",
    "express": "expressjs",
    "express.js": "expressjs",
    "next": "nextjs",
    "next.js": "nextjs",
    "scikit-learn": "sklearn",
    "sci-kit learn": "sklearn",
    "dotnet": ".net",
    "dot net": ".net",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "devops": "dev ops",
    "ui/ux": "ui ux",
    "ux/ui": "ui ux",
    "hr": "human resources",
    "crm": "customer relationship management",
    "erp": "enterprise resource planning",
    "sap": "sap erp",
    "roi": "return on investment",
    "kpi": "key performance indicator",
    "p&l": "profit and loss",
    "ms excel": "microsoft excel",
    "excel": "microsoft excel",
    "ms word": "microsoft word",
    "powerpoint": "microsoft powerpoint",
    "ppt": "microsoft powerpoint",
}


def _expand_skill(skill: str) -> set:
    """Return a set of normalized forms for a skill (original + synonym expansions)."""
    s = normalize_skill(skill)
    forms = {s}
    # Direct synonym lookup
    if s in SKILL_SYNONYMS:
        forms.add(normalize_skill(SKILL_SYNONYMS[s]))
    # Reverse lookup — if s is a synonym value, add corresponding keys
    for k, v in SKILL_SYNONYMS.items():
        if normalize_skill(v) == s:
            forms.add(normalize_skill(k))
    # Strip trailing version numbers (e.g., "python 3.9" → "python")
    base = re.sub(r'\s*[\d]+(\.[\d]+)*\s*$', '', s).strip()
    if base and base != s:
        forms.add(base)
    # Strip common suffixes like .js
    no_dot = s.replace('.js', 'js').replace('.net', 'dotnet')
    if no_dot != s:
        forms.add(no_dot)
    return forms


def _token_overlap_ratio(s1: str, s2: str) -> float:
    """Compute Jaccard-like token overlap ratio, ignoring generic modifiers."""
    words1 = set(s1.split()) - GENERIC_SKILL_MODIFIERS
    words2 = set(s2.split()) - GENERIC_SKILL_MODIFIERS
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def _char_similarity(s1: str, s2: str) -> float:
    """SequenceMatcher ratio for fuzzy character-level matching."""
    return SequenceMatcher(None, s1, s2).ratio()


# Pairs of skills that look similar (substring/fuzzy) but are completely distinct.
# Each tuple (a, b) means: a must NEVER match b.
CONFUSED_SKILL_PAIRS = {
    ("java", "javascript"), ("java", "javafx"),
    ("c", "c++"), ("c", "c#"), ("c", "css"), ("c", "csv"),
    ("c++", "c#"),
    ("r", "rust"), ("r", "ruby"),
    ("go", "google"),
    ("aws", "awscli"),
    ("mysql", "postgresql"), ("mysql", "mssql"),
    ("react", "react native"),
    ("angular", "angularjs"),
    ("swift", "swiftui"),
    ("typescript", "javascript"),
}


def _is_confused_pair(s1: str, s2: str) -> bool:
    """Check if two normalized skills are a known confused pair."""
    return (s1, s2) in CONFUSED_SKILL_PAIRS or (s2, s1) in CONFUSED_SKILL_PAIRS


def smart_skill_match(skill1: str, skill2: str) -> bool:
    """
    Fuzzy skill matching that handles abbreviations, synonyms, and close variations
    while preventing false positives from generic word overlap.

    Examples:
    - "JavaScript" matches "JS" (synonym)
    - "React.js" matches "ReactJS" (variation)
    - "Python 3.9" matches "Python" (version stripped)
    - "HR Operations" does NOT match "B2B Operations" (different domains)
    - "Project Management" matches "Project Management" (exact)
    - "Java" does NOT match "JavaScript" (confused pair)
    """
    s1 = normalize_skill(skill1)
    s2 = normalize_skill(skill2)

    # 0. Block known confused pairs early
    if _is_confused_pair(s1, s2):
        return False

    # 1. Exact match
    if s1 == s2:
        return True

    # 2. Synonym / abbreviation expansion match
    forms1 = _expand_skill(skill1)
    forms2 = _expand_skill(skill2)
    if forms1 & forms2:  # any shared normalized form
        return True

    # 3. Domain-aware guard for compound skills
    if is_compound_skill(skill1) or is_compound_skill(skill2):
        domain1 = get_skill_domain(skill1)
        domain2 = get_skill_domain(skill2)
        if domain1 and domain2 and domain1 != domain2:
            return False
        if (domain1 and not domain2) or (domain2 and not domain1):
            non_domain_skill = s2 if domain1 else s1
            if non_domain_skill in GENERIC_SKILL_MODIFIERS:
                return False

    # 4. Substring containment (e.g., "Python" in "Python Programming")
    #    Only match if the shorter string appears as a whole word in the longer one.
    if s1 in s2 or s2 in s1:
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        longer_words = set(longer.split())
        shorter_words = set(shorter.split())
        # Check if all words of the shorter skill appear as whole words in the longer
        if shorter_words.issubset(longer_words):
            non_generic_overlap = shorter_words - GENERIC_SKILL_MODIFIERS
            if non_generic_overlap:
                return True
            if shorter_words.issubset(GENERIC_SKILL_MODIFIERS):
                return False

    # 5. Token overlap ratio (e.g., "Machine Learning" vs "Machine Learning Engineering")
    if _token_overlap_ratio(s1, s2) >= 0.5:
        return True

    # 6. Character-level fuzzy match for close variations (e.g., "PostgreSQL" vs "Postgresql")
    if len(s1) >= 4 and len(s2) >= 4 and _char_similarity(s1, s2) >= 0.82:
        return True

    return False


# =============================================================================
# Google Drive Storage
# =============================================================================

class GoogleDriveClient:
    """Google Drive client for file storage using service account."""

    def __init__(self):
        if not all([GOOGLE_SHEETS_CREDENTIALS_FILE, GOOGLE_DRIVE_FOLDER_ID]):
            logger.warning("Google Drive credentials not configured")
            self.service = None
            self.folder_id = None
            return

        try:
            from googleapiclient.discovery import build
            from google.oauth2.service_account import Credentials as ServiceAccountCredentials

            credentials = ServiceAccountCredentials.from_service_account_file(
                GOOGLE_SHEETS_CREDENTIALS_FILE,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            self.service = build('drive', 'v3', credentials=credentials)
            self.folder_id = GOOGLE_DRIVE_FOLDER_ID
            logger.info("Google Drive client initialized for folder: %s", self.folder_id)
        except Exception as e:
            logger.error("Failed to initialize Google Drive client: %s", e)
            self.service = None
            self.folder_id = None

    def is_configured(self) -> bool:
        return self.service is not None

    def upload_file(self, file_data: bytes, filename: str, content_type: str = None) -> Optional[str]:
        """Upload file to Google Drive and return file ID."""
        if not self.is_configured():
            return None

        try:
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }

            from googleapiclient.http import MediaIoBaseUpload
            import io

            media = MediaIoBaseUpload(
                io.BytesIO(file_data),
                mimetype=content_type or 'application/octet-stream'
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            file_id = file.get('id')
            logger.info("Uploaded to Google Drive: %s (ID: %s)", filename, file_id)
            return file_id
        except Exception as e:
            logger.error("Google Drive upload failed for %s: %s", filename, e)
            return None

    def get_file(self, file_id: str) -> Optional[bytes]:
        """Download file from Google Drive."""
        if not self.is_configured():
            return None

        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            request = self.service.files().get_media(fileId=file_id)
            file_data = io.BytesIO()
            downloader = MediaIoBaseDownload(file_data, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            return file_data.getvalue()
        except Exception as e:
            logger.error("Google Drive download failed for %s: %s", file_id, e)
            return None

    def delete_file(self, file_id: str) -> bool:
        """Delete file from Google Drive."""
        if not self.is_configured():
            return False

        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info("Deleted from Google Drive: %s", file_id)
            return True
        except Exception as e:
            logger.error("Google Drive delete failed for %s: %s", file_id, e)
            return False

    def get_download_url(self, file_id: str) -> str:
        """Get direct download URL for file (requires authentication)."""
        return f"https://drive.google.com/uc?export=download&id={file_id}"


class SupabaseClient:
    """Supabase Storage client for file upload/download."""

    def __init__(self):
        if not all([SUPABASE_URL, SUPABASE_ANON_KEY]):
            logger.warning("Supabase credentials not configured")
            self.client = None
            self.bucket_name = None
            return

        try:
            from supabase import create_client
            self.client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            self.bucket_name = SUPABASE_BUCKET_NAME
            logger.info("Supabase client initialized for bucket: %s", self.bucket_name)
        except Exception as e:
            logger.error("Failed to initialize Supabase client: %s", e)
            self.client = None
            self.bucket_name = None

    def is_configured(self) -> bool:
        return self.client is not None

    def upload_file(self, file_data: bytes, filename: str, content_type: str = None) -> Optional[str]:
        """Upload file to Supabase Storage and return storage path."""
        if not self.is_configured():
            return None

        try:
            import time
            # Add timestamp to filename to avoid conflicts
            timestamp = int(time.time())
            storage_path = f"{timestamp}_{filename}"

            self.client.storage.from_(self.bucket_name).upload(
                path=storage_path,
                file=file_data,
                file_options={"content-type": content_type or "application/octet-stream"}
            )

            logger.info("Uploaded to Supabase: %s (path: %s)", filename, storage_path)
            return storage_path
        except Exception as e:
            logger.error("Supabase upload failed for %s: %s", filename, e)
            return None

    def get_file(self, storage_path: str) -> Optional[bytes]:
        """Download file from Supabase Storage."""
        if not self.is_configured():
            return None

        try:
            response = self.client.storage.from_(self.bucket_name).download(storage_path)
            return response
        except Exception as e:
            logger.error("Supabase download failed for %s: %s", storage_path, e)
            return None

    def get_download_url(self, storage_path: str) -> str:
        """Get signed download URL for file (valid for 1 hour)."""
        if not self.is_configured():
            return ""
        try:
            # Generate signed URL valid for 3600 seconds (1 hour)
            url = self.client.storage.from_(self.bucket_name).create_signed_url(
                storage_path,
                expires_in=3600
            )
            return url
        except Exception as e:
            logger.error("Failed to generate signed URL for %s: %s", storage_path, e)
            return ""


# =============================================================================
# Embedding & Matching
# =============================================================================

class EmbeddingMatcher:
    """Computes embeddings using OpenAI API to save memory (no local model)."""

    def __init__(self):
        self.api_key = LLM_API_KEY if LLM_PROVIDER == "openai" else None

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        text = (text or "").strip()
        if not text:
            return np.zeros((1536,), dtype=np.float32)  # OpenAI embedding dimension

        if not self.api_key:
            # Fallback: return zero embedding if no API key
            return np.zeros((1536,), dtype=np.float32)

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"input": text, "model": "text-embedding-3-small"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data["data"][0]["embedding"]
                    return np.asarray(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning("OpenAI embedding failed: %s", e)

        return np.zeros((1536,), dtype=np.float32)

    def embed_resume(self, resume_data: ResumeData, raw_text: str) -> np.ndarray:
        """Create embedding from resume data + raw text."""
        parts = [resume_data.skills_for_embedding()]
        if resume_data.summary:
            parts.append(resume_data.summary)
        parts.append(raw_text[:3000])
        return self.embed("\n".join(parts))

    def embed_job(self, job_data: JobData, title: str, description: str) -> np.ndarray:
        """Create embedding from job data + raw text."""
        parts = [title, job_data.skills_for_embedding()]
        if job_data.summary:
            parts.append(job_data.summary)
        parts.append(description[:3000])
        return self.embed("\n".join(parts))

    @staticmethod
    def embedding_to_json(emb: np.ndarray) -> str:
        return json.dumps(emb.astype(float).tolist())

    @staticmethod
    def embedding_from_json(value: str) -> np.ndarray:
        arr = json.loads(value)
        return np.asarray(arr, dtype=np.float32)

    def rank_matches(self, query_emb: np.ndarray, rows: List[sqlite3.Row], emb_field: str, top_k: int = 10):
        """Rank rows by cosine similarity to query embedding."""
        scored = []
        for r in rows:
            emb = self.embedding_from_json(r[emb_field])
            score = float(np.dot(query_emb, emb))
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# =============================================================================
# Vector Store (ChromaDB)
# =============================================================================

class SupabaseVectorStore:
    """Supabase pgvector-backed vector store for embedding storage and similarity search."""

    def __init__(self, db_url: str = SUPABASE_DB_URL):
        self.db_url = db_url

    def _get_conn(self):
        import psycopg2
        # Add SSL mode if not specified in connection string
        db_url = self.db_url
        if "sslmode" not in db_url.lower():
            db_url += "&sslmode=require" if "?" in db_url else "?sslmode=require"
        # Force IPv4 by adding host parameter to avoid IPv6 connection issues
        import re
        match = re.search(r'@([^:/]+)', db_url)
        if match and "host=" not in db_url.lower():
            host = match.group(1)
            db_url += f"&host={host}" if "?" in db_url else f"?host={host}"
        return psycopg2.connect(db_url)

    def add_resume(self, resume_id: int, embedding: np.ndarray) -> None:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO resume_embeddings (id, embedding) VALUES (%s, %s)
                   ON CONFLICT (id) DO UPDATE SET embedding = %s""",
                (resume_id, embedding.tolist(), embedding.tolist())
            )
            conn.commit()
        finally:
            conn.close()

    def add_job(self, job_id: int, embedding: np.ndarray) -> None:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO job_embeddings (id, embedding) VALUES (%s, %s)
                   ON CONFLICT (id) DO UPDATE SET embedding = %s""",
                (job_id, embedding.tolist(), embedding.tolist())
            )
            conn.commit()
        finally:
            conn.close()

    def get_resume_embedding(self, resume_id: int) -> Optional[np.ndarray]:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM resume_embeddings WHERE id = %s", (resume_id,))
            result = cursor.fetchone()
            if result:
                return np.asarray(result[0], dtype=np.float32)
            return None
        finally:
            conn.close()

    def get_job_embedding(self, job_id: int) -> Optional[np.ndarray]:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM job_embeddings WHERE id = %s", (job_id,))
            result = cursor.fetchone()
            if result:
                return np.asarray(result[0], dtype=np.float32)
            return None
        finally:
            conn.close()

    def query_similar_jobs(self, query_embedding: np.ndarray, top_k: int = 50) -> dict:
        """Query top-k most similar jobs using pgvector cosine similarity."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, 1 - (embedding <=> %s::vector) as similarity
                   FROM job_embeddings
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (query_embedding.tolist(), query_embedding.tolist(), top_k)
            )
            results = cursor.fetchall()
            return {row[0]: max(0.0, min(1.0, row[1])) for row in results}
        finally:
            conn.close()

    def query_similar_resumes(self, query_embedding: np.ndarray, top_k: int = 50) -> dict:
        """Query top-k most similar resumes using pgvector cosine similarity."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, 1 - (embedding <=> %s::vector) as similarity
                   FROM resume_embeddings
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (query_embedding.tolist(), query_embedding.tolist(), top_k)
            )
            results = cursor.fetchall()
            return {row[0]: max(0.0, min(1.0, row[1])) for row in results}
        finally:
            conn.close()

    def delete_jobs(self, job_ids: List[int]) -> None:
        if job_ids:
            import psycopg2
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM job_embeddings WHERE id = ANY(%s)",
                    (job_ids,)
                )
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def delete_job(self, job_id: int) -> None:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM job_embeddings WHERE id = %s", (job_id,))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    def delete_resume(self, resume_id: int) -> None:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM resume_embeddings WHERE id = %s", (resume_id,))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


class VectorStore:
    """ChromaDB-backed vector store for embedding storage and similarity search."""

    def __init__(self, path: str = CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.resumes = self.client.get_or_create_collection(
            "resumes", metadata={"hnsw:space": "cosine"})
        self.jobs = self.client.get_or_create_collection(
            "jobs", metadata={"hnsw:space": "cosine"})

    def add_resume(self, resume_id: int, embedding: np.ndarray) -> None:
        self.resumes.upsert(
            ids=[str(resume_id)],
            embeddings=[embedding.astype(float).tolist()])

    def add_job(self, job_id: int, embedding: np.ndarray) -> None:
        self.jobs.upsert(
            ids=[str(job_id)],
            embeddings=[embedding.astype(float).tolist()])

    def get_resume_embedding(self, resume_id: int) -> Optional[np.ndarray]:
        result = self.resumes.get(ids=[str(resume_id)], include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return np.asarray(embeddings[0], dtype=np.float32)
        return None

    def get_job_embedding(self, job_id: int) -> Optional[np.ndarray]:
        result = self.jobs.get(ids=[str(job_id)], include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings and len(embeddings) > 0:
            return np.asarray(embeddings[0], dtype=np.float32)
        return None

    def query_similar_jobs(self, query_embedding: np.ndarray, top_k: int = 50) -> dict:
        """Query top-k most similar jobs. Returns {job_id: similarity_score}."""
        count = self.jobs.count()
        if count == 0:
            return {}
        n = min(top_k, count)
        results = self.jobs.query(
            query_embeddings=[query_embedding.astype(float).tolist()],
            n_results=n)
        scores = {}
        for id_str, dist in zip(results["ids"][0], results["distances"][0]):
            scores[int(id_str)] = max(0.0, min(1.0, 1.0 - dist))
        return scores

    def query_similar_resumes(self, query_embedding: np.ndarray, top_k: int = 50) -> dict:
        """Query top-k most similar resumes. Returns {resume_id: similarity_score}."""
        count = self.resumes.count()
        if count == 0:
            return {}
        n = min(top_k, count)
        results = self.resumes.query(
            query_embeddings=[query_embedding.astype(float).tolist()],
            n_results=n)
        scores = {}
        for id_str, dist in zip(results["ids"][0], results["distances"][0]):
            scores[int(id_str)] = max(0.0, min(1.0, 1.0 - dist))
        return scores

    def delete_jobs(self, job_ids: List[int]) -> None:
        if job_ids:
            try:
                self.jobs.delete(ids=[str(jid) for jid in job_ids])
            except Exception:
                pass

    def delete_resume(self, resume_id: int) -> None:
        try:
            self.resumes.delete(ids=[str(resume_id)])
        except Exception:
            pass


# =============================================================================
# Database
# =============================================================================

class SupabaseDatabase:
    """Supabase PostgreSQL database for storing resumes and jobs."""

    def __init__(self, db_url: str = SUPABASE_DB_URL, vector_store: 'VectorStore' = None):
        self.db_url = db_url
        self.vector_store = vector_store
        self._init_db()

    def _get_conn(self):
        import psycopg2
        # Add SSL mode if not specified in connection string
        db_url = self.db_url
        if "sslmode" not in db_url.lower():
            db_url += "&sslmode=require" if "?" in db_url else "?sslmode=require"
        # Force IPv4 by adding host parameter to avoid IPv6 connection issues
        import re
        match = re.search(r'@([^:/]+)', db_url)
        if match and "host=" not in db_url.lower():
            host = match.group(1)
            db_url += f"&host={host}" if "?" in db_url else f"?host={host}"
        return psycopg2.connect(db_url)

    def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        # Tables are created manually in Supabase SQL Editor
        # This method can be used for migrations if needed
        pass

    def insert_resume(self, filename: str, content_type: str, raw_text: str,
                      extracted_data: ResumeData, embedding: np.ndarray, file_path: str, storage_type: str = "local") -> int:
        """Insert or update resume. If filename exists, override it."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            # Check if resume with same filename exists
            cursor.execute(
                "SELECT id FROM resumes WHERE filename = %s", (filename,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing resume
                cursor.execute(
                    """UPDATE resumes SET content_type=%s, raw_text=%s, extracted_data_json=%s,
                       embedding_json=%s, file_path=%s, storage_type=%s, created_at=%s WHERE filename=%s""",
                    (content_type, raw_text, extracted_data.to_json(),
                     EmbeddingMatcher.embedding_to_json(embedding), file_path, storage_type,
                     datetime.now(timezone.utc).isoformat(), filename)
                )
                conn.commit()
                resume_id = existing[0]
            else:
                # Insert new resume
                cursor.execute(
                    """INSERT INTO resumes (filename, content_type, raw_text, extracted_data_json, embedding_json, file_path, storage_type, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (filename, content_type, raw_text, extracted_data.to_json(),
                     EmbeddingMatcher.embedding_to_json(embedding), file_path, storage_type,
                     datetime.now(timezone.utc).isoformat())
                )
                resume_id = cursor.fetchone()[0]
            conn.commit()
            # Store embedding in vector store
            if self.vector_store:
                self.vector_store.add_resume(resume_id, embedding)
            return resume_id
        finally:
            conn.close()

    def get_all_resumes(self) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM resumes ORDER BY id DESC")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_resume_by_filename(self, filename: str) -> Optional[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM resumes WHERE filename = %s", (filename,))
            return cursor.fetchone()
        finally:
            conn.close()

    def insert_job(self, title: str, description: str, extracted_data: JobData, embedding: np.ndarray,
                   posted_date: str = None, source: str = "manual", posted_by: str = "") -> int:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO jobs (title, description, extracted_data_json, embedding_json, posted_date, source, posted_by, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (title, description, extracted_data.to_json(),
                 EmbeddingMatcher.embedding_to_json(embedding),
                 posted_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                 source,
                 posted_by,
                 datetime.now(timezone.utc).isoformat())
            )
            job_id = cursor.fetchone()[0]
            conn.commit()
            # Store embedding in vector store
            if self.vector_store:
                self.vector_store.add_job(job_id, embedding)
            return job_id
        finally:
            conn.close()

    def job_exists_by_title_and_source(self, title: str, source: str = "google_sheet") -> bool:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM jobs WHERE title = %s AND source = %s",
                (title, source)
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def track_skills(self, skills_by_category: dict, source: str = "resume") -> None:
        """Track skills in the taxonomy. source is 'resume' or 'job'."""
        import psycopg2
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        skill_ids = []
        
        try:
            for category, skills in skills_by_category.items():
                for skill in skills:
                    if not skill or not skill.strip():
                        continue
                    skill_name = skill.strip()
                    skill_lower = skill_name.lower()
                    
                    # Try to insert or update skill
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id FROM skills WHERE skill_name_lower = %s",
                        (skill_lower,)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        skill_id = existing[0]
                        if source == "resume":
                            cursor.execute(
                                "UPDATE skills SET resume_count = resume_count + 1, last_seen = %s WHERE id = %s",
                                (now, skill_id)
                            )
                        else:
                            cursor.execute(
                                "UPDATE skills SET job_count = job_count + 1, last_seen = %s WHERE id = %s",
                                (now, skill_id)
                            )
                    else:
                        cursor.execute(
                            """INSERT INTO skills (skill_name, skill_name_lower, category, source, resume_count, job_count, first_seen, last_seen)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                            (skill_name, skill_lower, category, source,
                             1 if source == "resume" else 0,
                             1 if source == "job" else 0,
                             now, now)
                        )
                        skill_id = cursor.fetchone()[0]
                    
                    skill_ids.append(skill_id)
            
            # Track co-occurrences (skills that appear together)
            for i, id1 in enumerate(skill_ids):
                for id2 in skill_ids[i+1:]:
                    # Ensure consistent ordering
                    s1, s2 = min(id1, id2), max(id1, id2)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT count FROM skill_cooccurrence WHERE skill1_id = %s AND skill2_id = %s",
                        (s1, s2)
                    )
                    existing = cursor.fetchone()
                    if existing:
                        cursor.execute(
                            "UPDATE skill_cooccurrence SET count = count + 1 WHERE skill1_id = %s AND skill2_id = %s",
                            (s1, s2)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO skill_cooccurrence (skill1_id, skill2_id, count) VALUES (%s, %s, 1)",
                            (s1, s2)
                        )
            
            conn.commit()
        finally:
            conn.close()

    def get_jobs(self) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_all_jobs(self) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM jobs ORDER BY id DESC")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_stats(self) -> Tuple[int, int]:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM resumes")
            resume_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM jobs")
            job_count = cursor.fetchone()[0]
            return (resume_count, job_count)
        finally:
            conn.close()

    def get_recent_resumes(self, limit: int = 10) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT * FROM resumes ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def get_recent_jobs(self, limit: int = 10) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def delete_job(self, job_id: int) -> bool:
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
            conn.commit()
            if self.vector_store:
                self.vector_store.delete_job(job_id)
            return True
        except Exception as e:
            logger.error("Failed to delete job: %s", e)
            return False
        finally:
            conn.close()

    def cleanup_expired_jobs(self, retention_days: int = 30) -> int:
        """Delete jobs older than retention_days. Returns count of deleted jobs."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            cursor.execute(
                "DELETE FROM jobs WHERE posted_date < %s RETURNING id",
                (cutoff.strftime("%Y-%m-%d"),)
            )
            deleted_ids = [row[0] for row in cursor.fetchall()]
            conn.commit()
            # Delete from vector store
            if self.vector_store:
                for job_id in deleted_ids:
                    self.vector_store.delete_job(job_id)
            return len(deleted_ids)
        finally:
            conn.close()

    def get_all_skills(self) -> List[dict]:
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute("SELECT * FROM skills ORDER BY skill_name")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_top_skills(self, limit: int = 20, source: str = "all") -> List[dict]:
        """Get most common skills."""
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if source == "all":
                cursor.execute(
                    "SELECT * FROM skills ORDER BY (resume_count + job_count) DESC LIMIT %s",
                    (limit,)
                )
            elif source == "resume":
                cursor.execute(
                    "SELECT * FROM skills ORDER BY resume_count DESC LIMIT %s",
                    (limit,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM skills ORDER BY job_count DESC LIMIT %s",
                    (limit,)
                )
            return cursor.fetchall()
        finally:
            conn.close()

    def get_related_skills(self, skill_name: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Get skills that frequently appear with the given skill."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT s1.skill_name, sc.count
                   FROM skill_cooccurrence sc
                   JOIN skills s1 ON sc.skill1_id = s1.id
                   JOIN skills s2 ON sc.skill2_id = s2.id
                   WHERE s2.skill_name_lower = %s OR s1.skill_name_lower = %s
                   ORDER BY sc.count DESC
                   LIMIT %s""",
                (skill_name.lower(), skill_name.lower(), limit)
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_skills_stats(self) -> dict:
        """Get overall skills statistics."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM skills")
            total_skills = cursor.fetchone()[0]
            cursor.execute("SELECT SUM(resume_count) FROM skills")
            total_resume_mentions = cursor.fetchone()[0] or 0
            cursor.execute("SELECT SUM(job_count) FROM skills")
            total_job_mentions = cursor.fetchone()[0] or 0
            return {
                "total_skills": total_skills,
                "total_resume_mentions": total_resume_mentions,
                "total_job_mentions": total_job_mentions
            }
        finally:
            conn.close()

    def get_skills_by_category(self) -> dict:
        """Get skill counts grouped by category."""
        import psycopg2
        import psycopg2.extras
        conn = self._get_conn()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                "SELECT category, COUNT(*) as count FROM skills GROUP BY category ORDER BY count DESC"
            )
            return {row["category"]: row["count"] for row in cursor.fetchall()}
        finally:
            conn.close()

    def update_skill_taxonomy(self, skills: List[str], category: str, source: str, 
                             resume_count: int = 0, job_count: int = 0) -> None:
        """Update skills taxonomy table with discovered skills."""
        import psycopg2
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            for skill in skills:
                skill_lower = skill.lower()
                cursor.execute(
                    """INSERT INTO skills (skill_name, skill_name_lower, category, source, resume_count, job_count, first_seen, last_seen)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (skill_name_lower) DO UPDATE SET
                           resume_count = skills.resume_count + %s,
                           job_count = skills.job_count + %s,
                           last_seen = %s""",
                    (skill, skill_lower, category, source, resume_count, job_count, now, now,
                     resume_count, job_count, now)
                )
            conn.commit()
        finally:
            conn.close()


class Database:
    """Database for storing resumes and jobs."""

    def __init__(self, db_path: str = DB_PATH, vector_store: 'VectorStore' = None):
        self.db_path = db_path
        self.vector_store = vector_store
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    content_type TEXT,
                    raw_text TEXT NOT NULL,
                    extracted_data_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    file_path TEXT,
                    storage_type TEXT DEFAULT 'local',
                    created_at TEXT NOT NULL
                );
            """)
            # Migration: Add storage_type column if not exists
            cursor = conn.execute("PRAGMA table_info(resumes)")
            columns = [row[1] for row in cursor.fetchall()]
            if "storage_type" not in columns:
                conn.execute("ALTER TABLE resumes ADD COLUMN storage_type TEXT DEFAULT 'local'")
                conn.commit()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    extracted_data_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    posted_date TEXT,
                    source TEXT DEFAULT 'manual',
                    posted_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
            """)
            # Skills taxonomy table - tracks all discovered skills
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL UNIQUE,
                    skill_name_lower TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    resume_count INTEGER DEFAULT 0,
                    job_count INTEGER DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
            """)
            # Skill co-occurrence table - tracks which skills appear together
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skill_cooccurrence (
                    skill1_id INTEGER NOT NULL,
                    skill2_id INTEGER NOT NULL,
                    count INTEGER DEFAULT 1,
                    PRIMARY KEY (skill1_id, skill2_id),
                    FOREIGN KEY (skill1_id) REFERENCES skills(id),
                    FOREIGN KEY (skill2_id) REFERENCES skills(id)
                );
            """)
            # Add file_path column if it doesn't exist (for existing DBs)
            try:
                conn.execute("ALTER TABLE resumes ADD COLUMN file_path TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Add posted_date and source columns to jobs if they don't exist (for existing DBs)
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN posted_date TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN source TEXT DEFAULT 'manual'")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN posted_by TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    def insert_resume(self, filename: str, content_type: str, raw_text: str,
                      extracted_data: ResumeData, embedding: np.ndarray, file_path: str, storage_type: str = "local") -> int:
        """Insert or update resume. If filename exists, override it."""
        conn = self._get_conn()
        try:
            # Check if resume with same filename exists
            existing = conn.execute(
                "SELECT id FROM resumes WHERE filename = ?", (filename,)
            ).fetchone()

            if existing:
                # Update existing resume
                conn.execute(
                    """UPDATE resumes SET content_type=?, raw_text=?, extracted_data_json=?,
                       embedding_json=?, file_path=?, storage_type=?, created_at=? WHERE filename=?""",
                    (content_type, raw_text, extracted_data.to_json(),
                     EmbeddingMatcher.embedding_to_json(embedding), file_path, storage_type,
                     datetime.now(timezone.utc).isoformat(), filename)
                )
                conn.commit()
                resume_id = existing["id"]
            else:
                # Insert new resume
                cursor = conn.execute(
                    """INSERT INTO resumes (filename, content_type, raw_text, extracted_data_json, embedding_json, file_path, storage_type, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (filename, content_type, raw_text, extracted_data.to_json(),
                     EmbeddingMatcher.embedding_to_json(embedding), file_path, storage_type,
                     datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
                resume_id = cursor.lastrowid
            # Store embedding in ChromaDB
            if self.vector_store:
                self.vector_store.add_resume(resume_id, embedding)
            return resume_id
        finally:
            conn.close()

    def insert_job(self, title: str, description: str, extracted_data: JobData, embedding: np.ndarray,
                   posted_date: str = None, source: str = "manual", posted_by: str = "") -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO jobs (title, description, extracted_data_json, embedding_json, posted_date, source, posted_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (title, description, extracted_data.to_json(),
                 EmbeddingMatcher.embedding_to_json(embedding),
                 posted_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                 source,
                 posted_by,
                 datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
            job_id = cursor.lastrowid
            # Store embedding in ChromaDB
            if self.vector_store:
                self.vector_store.add_job(job_id, embedding)
            return job_id
        finally:
            conn.close()

    def job_exists_by_title_and_source(self, title: str, source: str = "google_sheet") -> bool:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM jobs WHERE title = ? AND source = ?", (title, source)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def cleanup_expired_jobs(self, retention_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            # Get IDs of jobs to delete (for ChromaDB cleanup)
            expired = conn.execute(
                "SELECT id FROM jobs WHERE posted_date < ?", (cutoff,)
            ).fetchall()
            expired_ids = [r["id"] for r in expired]
            if expired_ids and self.vector_store:
                self.vector_store.delete_jobs(expired_ids)
            cursor = conn.execute(
                "DELETE FROM jobs WHERE posted_date < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def search_jobs(self, query: str) -> List[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute(
                "SELECT * FROM jobs WHERE title LIKE ? OR description LIKE ? ORDER BY id DESC",
                (f"%{query}%", f"%{query}%")
            ).fetchall()
        finally:
            conn.close()

    def get_all_jobs(self) -> List[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
        finally:
            conn.close()

    def get_all_resumes(self) -> List[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT * FROM resumes ORDER BY id DESC").fetchall()
        finally:
            conn.close()

    def get_resume_by_filename(self, filename: str) -> Optional[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT * FROM resumes WHERE filename = ?", (filename,)).fetchone()
        finally:
            conn.close()

    def get_stats(self) -> Tuple[int, int]:
        conn = self._get_conn()
        try:
            resume_count = conn.execute("SELECT COUNT(*) AS c FROM resumes").fetchone()["c"]
            job_count = conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
            return resume_count, job_count
        finally:
            conn.close()

    def get_recent_resumes(self, limit: int = 10) -> List[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute(
                "SELECT id, filename, extracted_data_json, created_at FROM resumes ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        finally:
            conn.close()

    def get_recent_jobs(self, limit: int = 10) -> List[sqlite3.Row]:
        conn = self._get_conn()
        try:
            return conn.execute(
                "SELECT id, title, extracted_data_json, created_at FROM jobs ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        finally:
            conn.close()

    def track_skills(self, skills_by_category: dict, source: str = "resume") -> None:
        """Track skills in the taxonomy. source is 'resume' or 'job'."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        skill_ids = []
        
        try:
            for category, skills in skills_by_category.items():
                for skill in skills:
                    if not skill or not skill.strip():
                        continue
                    skill_name = skill.strip()
                    skill_lower = skill_name.lower()
                    
                    # Try to insert or update skill
                    existing = conn.execute(
                        "SELECT id FROM skills WHERE skill_name_lower = ?", (skill_lower,)
                    ).fetchone()
                    
                    if existing:
                        skill_id = existing["id"]
                        if source == "resume":
                            conn.execute(
                                "UPDATE skills SET resume_count = resume_count + 1, last_seen = ? WHERE id = ?",
                                (now, skill_id)
                            )
                        else:
                            conn.execute(
                                "UPDATE skills SET job_count = job_count + 1, last_seen = ? WHERE id = ?",
                                (now, skill_id)
                            )
                    else:
                        cursor = conn.execute(
                            """INSERT INTO skills (skill_name, skill_name_lower, category, source, resume_count, job_count, first_seen, last_seen)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (skill_name, skill_lower, category, source,
                             1 if source == "resume" else 0,
                             1 if source == "job" else 0,
                             now, now)
                        )
                        skill_id = cursor.lastrowid
                    
                    skill_ids.append(skill_id)
            
            # Track co-occurrences (skills that appear together)
            for i, id1 in enumerate(skill_ids):
                for id2 in skill_ids[i+1:]:
                    # Ensure consistent ordering
                    s1, s2 = min(id1, id2), max(id1, id2)
                    existing = conn.execute(
                        "SELECT count FROM skill_cooccurrence WHERE skill1_id = ? AND skill2_id = ?",
                        (s1, s2)
                    ).fetchone()
                    if existing:
                        conn.execute(
                            "UPDATE skill_cooccurrence SET count = count + 1 WHERE skill1_id = ? AND skill2_id = ?",
                            (s1, s2)
                        )
                    else:
                        conn.execute(
                            "INSERT INTO skill_cooccurrence (skill1_id, skill2_id, count) VALUES (?, ?, 1)",
                            (s1, s2)
                        )
            
            conn.commit()
        finally:
            conn.close()

    def get_top_skills(self, limit: int = 20, source: str = "all") -> List[sqlite3.Row]:
        """Get most common skills."""
        conn = self._get_conn()
        try:
            if source == "resume":
                return conn.execute(
                    "SELECT * FROM skills WHERE resume_count > 0 ORDER BY resume_count DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            elif source == "job":
                return conn.execute(
                    "SELECT * FROM skills WHERE job_count > 0 ORDER BY job_count DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                return conn.execute(
                    "SELECT *, (resume_count + job_count) as total_count FROM skills ORDER BY total_count DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        finally:
            conn.close()

    def get_related_skills(self, skill_name: str, limit: int = 10) -> List[Tuple[str, int]]:
        """Get skills that frequently appear with the given skill."""
        conn = self._get_conn()
        try:
            # Find the skill ID
            skill = conn.execute(
                "SELECT id FROM skills WHERE skill_name_lower = ?", (skill_name.lower(),)
            ).fetchone()
            if not skill:
                return []
            
            skill_id = skill["id"]
            # Find co-occurring skills
            results = conn.execute("""
                SELECT s.skill_name, c.count
                FROM skill_cooccurrence c
                JOIN skills s ON (
                    CASE WHEN c.skill1_id = ? THEN c.skill2_id ELSE c.skill1_id END = s.id
                )
                WHERE c.skill1_id = ? OR c.skill2_id = ?
                ORDER BY c.count DESC
                LIMIT ?
            """, (skill_id, skill_id, skill_id, limit)).fetchall()
            
            return [(r["skill_name"], r["count"]) for r in results]
        finally:
            conn.close()

    def get_skills_stats(self) -> dict:
        """Get overall skills statistics."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) as c FROM skills").fetchone()["c"]
            from_resumes = conn.execute("SELECT COUNT(*) as c FROM skills WHERE resume_count > 0").fetchone()["c"]
            from_jobs = conn.execute("SELECT COUNT(*) as c FROM skills WHERE job_count > 0").fetchone()["c"]
            return {
                "total_unique_skills": total,
                "skills_from_resumes": from_resumes,
                "skills_from_jobs": from_jobs
            }
        finally:
            conn.close()

    def get_skills_by_category(self) -> dict:
        """Get skill counts grouped by category."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT category, COUNT(*) as count, SUM(resume_count) as resume_total, SUM(job_count) as job_total FROM skills GROUP BY category ORDER BY count DESC"
            ).fetchall()
            return {r["category"]: {"count": r["count"], "resume_total": r["resume_total"], "job_total": r["job_total"]} for r in rows}
        finally:
            conn.close()


# =============================================================================
# HTML Templates
# =============================================================================

def html_page(title: str, body: str, error: str = None) -> str:
    error_html = f"<div class='error'>{error}</div>" if error else ""
    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; max-width: 1100px; }}
      h1 {{ margin-bottom: 8px; color: #111827; }}
      h2 {{ color: #374151; margin-top: 24px; }}
      h3 {{ color: #4b5563; margin-top: 16px; margin-bottom: 8px; font-size: 0.95em; }}
      .nav {{ margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb; }}
      .nav a {{ margin-right: 16px; color: #2563eb; text-decoration: none; }}
      .nav a:hover {{ text-decoration: underline; }}
      .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin: 12px 0; background: #fafafa; }}
      .card-header {{ font-weight: 600; font-size: 1.1em; margin-bottom: 8px; }}
      .muted {{ color: #6b7280; font-size: 0.9em; }}
      .score {{ font-variant-numeric: tabular-nums; color: #059669; font-weight: 600; }}
      .skills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
      .skill-tag {{ background: #dbeafe; color: #1e40af; padding: 4px 10px; border-radius: 12px; font-size: 0.8em; }}
      .skill-tag.technical {{ background: #dbeafe; color: #1e40af; }}
      .skill-tag.business {{ background: #fef3c7; color: #92400e; }}
      .skill-tag.domain {{ background: #dcfce7; color: #166534; }}
      .skill-tag.soft {{ background: #f3e8ff; color: #7c3aed; }}
      .skill-tag.cert {{ background: #fee2e2; color: #dc2626; }}
      .skill-tag.tool {{ background: #e0e7ff; color: #4338ca; }}
      .skill-tag.required {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
      .skill-tag.preferred {{ background: #fef3c7; color: #92400e; }}
      .skill-section {{ margin-top: 10px; }}
      .skill-label {{ font-size: 0.85em; color: #6b7280; margin-bottom: 4px; }}
      textarea {{ width: 100%; min-height: 200px; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-family: inherit; }}
      input[type='text'] {{ width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; }}
      input[type='file'] {{ margin: 10px 0; }}
      button {{ padding: 12px 20px; border-radius: 8px; border: none; background: #2563eb; color: white; cursor: pointer; font-size: 1em; }}
      button:hover {{ background: #1d4ed8; }}
      .error {{ color: #dc2626; background: #fef2f2; padding: 12px; border-radius: 8px; margin: 12px 0; }}
      .success {{ color: #166534; background: #dcfce7; padding: 12px; border-radius: 8px; margin: 12px 0; }}
      .warning {{ color: #92400e; background: #fef3c7; padding: 12px; border-radius: 8px; margin: 12px 0; }}
      pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 0.85em; }}
      .summary {{ font-style: italic; color: #4b5563; margin-top: 8px; }}
      .status {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
      .status.online {{ background: #dcfce7; color: #166534; }}
      .status.offline {{ background: #fef2f2; color: #dc2626; }}
      .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 8px; }}
      .badge.industry {{ background: #e0e7ff; color: #4338ca; }}
      .badge.seniority {{ background: #fce7f3; color: #be185d; }}
      .meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 0.85em; color: #6b7280; }}
    </style>
  </head>
  <body>
    <div class="nav">
      <a href="/">Home</a>
      <a href="/jobs">Jobs Board</a>
      <a href="/candidate">Candidate Portal</a>
      <a href="/recruiter">Recruiter Portal</a>
      <a href="/admin">Admin</a>
    </div>
    {error_html}
    {body}
  </body>
</html>"""


def render_skills_section(label: str, skills: List[str], css_class: str = "") -> str:
    if not skills:
        return ""
    tags = "".join([f"<span class='skill-tag {css_class}'>{s}</span>" for s in skills])
    return f"""<div class='skill-section'>
        <div class='skill-label'>{label}</div>
        <div class='skills'>{tags}</div>
    </div>"""


def render_skills_tags(skills: List[str], css_class: str = "") -> str:
    if not skills:
        return "<span class='muted'>None</span>"
    return "".join([f"<span class='skill-tag {css_class}'>{s}</span>" for s in skills])


def snippet(text: str, max_len: int = 300) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[:max_len].rstrip() + "..."


def compute_skill_matches(resume_data: ResumeData, job_data: JobData, job_title: str = "") -> dict:
    """Compute which skills from resume match job requirements. Returns skill-based match score."""
    # Find matches using smart skill matching
    matched_required = []
    matched_preferred = []
    missing_required = []
    matched_domain = []
    
    resume_skills = resume_data.all_skills()
    
    # Check required skills with smart matching
    for job_skill in job_data.all_required_skills():
        found = False
        for resume_skill in resume_skills:
            if smart_skill_match(job_skill, resume_skill):
                matched_required.append(job_skill)
                found = True
                break
        if not found:
            missing_required.append(job_skill)
    
    # Check preferred skills with smart matching
    for job_skill in job_data.all_preferred_skills():
        for resume_skill in resume_skills:
            if smart_skill_match(job_skill, resume_skill):
                matched_preferred.append(job_skill)
                break
    
    # Check domain/industry match (for jobs with minimal skill requirements)
    # Match job industry/department/title against resume's domain expertise, industries, and job titles
    job_domain_terms = []
    if job_data.industry:
        job_domain_terms.extend([t.strip().lower() for t in job_data.industry.split(',')])
    if job_data.department:
        job_domain_terms.extend([t.strip().lower() for t in job_data.department.split(',')])
    # Also extract keywords from job title for domain matching
    if job_title:
        # Split job title into words and use meaningful ones (length >= 2)
        title_words = [w.lower().strip() for w in job_title.replace('-', ' ').replace('/', ' ').split() if len(w) >= 2]
        job_domain_terms.extend(title_words)
    
    resume_domain_terms = []
    resume_domain_terms.extend([d.lower() for d in resume_data.domain_expertise])
    resume_domain_terms.extend([i.lower() for i in resume_data.industries])
    resume_domain_terms.extend([j.lower() for j in resume_data.job_titles])
    
    for job_term in job_domain_terms:
        if not job_term:
            continue
        for res_term in resume_domain_terms:
            # Use smart matching for domain terms too
            if smart_skill_match(job_term, res_term):
                matched_domain.append(job_term.title())
                break
    
    # Calculate skill-based match score
    # Skills are MANDATORY for matching; domain is OPTIONAL (small bonus only)
    total_required = len(job_data.all_required_skills())
    total_preferred = len(job_data.all_preferred_skills())
    total_domain = len(job_domain_terms) if job_domain_terms else 0
    
    required_score = len(matched_required) / total_required if total_required > 0 else 0
    preferred_score = len(matched_preferred) / total_preferred if total_preferred > 0 else 0
    domain_score = len(matched_domain) / total_domain if total_domain > 0 else 0
    
    # Weighted score calculation — skills drive the score, domain is optional bonus
    # Domain can add up to 10% bonus on top of skill scores
    domain_bonus = domain_score * 0.1 if total_domain > 0 else 0
    
    if total_required > 0 and total_preferred > 0:
        match_score = (required_score * 0.7) + (preferred_score * 0.3) + domain_bonus
    elif total_required > 0:
        match_score = required_score + domain_bonus
    elif total_preferred > 0:
        match_score = preferred_score + domain_bonus
    else:
        # No skills to match — score is 0 (domain alone cannot drive matching)
        match_score = 0
    
    # Clamp to [0, 1]
    match_score = min(match_score, 1.0)
    
    return {
        "matched_required": matched_required,
        "matched_preferred": matched_preferred,
        "missing_required": missing_required,
        "matched_domain": matched_domain,
        "match_score": match_score,  # 0 to 1
        "match_percentage": match_score * 100
    }


def rank_by_skills(resume_data: ResumeData, jobs: list,
                   vector_store: 'VectorStore' = None,
                   resume_embedding: np.ndarray = None) -> list:
    """Rank jobs by blended score: 40% skill match + 60% semantic similarity."""
    # Pre-fetch semantic scores from ChromaDB in a single query
    semantic_scores = {}
    if vector_store is not None and resume_embedding is not None:
        semantic_scores = vector_store.query_similar_jobs(resume_embedding, top_k=max(len(jobs), 1))

    scored = []
    for job in jobs:
        job_data = JobData.from_json(job["extracted_data_json"])
        match_info = compute_skill_matches(resume_data, job_data, job["title"])
        skill_score = match_info["match_score"]

        # Semantic similarity from ChromaDB
        semantic_score = semantic_scores.get(job["id"], 0.0)
        if semantic_scores:
            blended = 0.4 * skill_score + 0.6 * semantic_score
        else:
            blended = skill_score
        match_info["semantic_score"] = semantic_score
        match_info["blended_score"] = blended
        scored.append((blended, job, match_info))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def rank_resumes_by_skills(job_data: JobData, resumes: list, job_title: str = "",
                           vector_store: 'VectorStore' = None,
                           job_embedding: np.ndarray = None) -> list:
    """Rank resumes by blended score: 40% skill match + 60% semantic similarity."""
    # Pre-fetch semantic scores from ChromaDB in a single query
    semantic_scores = {}
    if vector_store is not None and job_embedding is not None:
        semantic_scores = vector_store.query_similar_resumes(job_embedding, top_k=max(len(resumes), 1))

    scored = []
    for res in resumes:
        resume_data = ResumeData.from_json(res["extracted_data_json"])
        match_info = compute_skill_matches(resume_data, job_data, job_title)
        skill_score = match_info["match_score"]

        # Semantic similarity from ChromaDB
        semantic_score = semantic_scores.get(res["id"], 0.0)
        if semantic_scores:
            blended = 0.4 * skill_score + 0.6 * semantic_score
        else:
            blended = skill_score
        match_info["semantic_score"] = semantic_score
        match_info["blended_score"] = blended
        scored.append((blended, res, match_info))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def render_match_analysis(match_info: dict) -> str:
    """Render the match analysis section."""
    html = "<div class='match-analysis' style='margin-top:12px; padding:10px; background:#f0fdf4; border-radius:8px; border:1px solid #86efac;'>"
    html += "<div style='font-weight:600; margin-bottom:8px;'>Match Analysis</div>"
    
    if match_info["matched_required"]:
        html += f"<div style='margin-bottom:6px;'><span style='color:#166534;'>✓ Matching Required Skills ({len(match_info['matched_required'])}):</span></div>"
        html += "<div class='skills'>" + "".join([f"<span class='skill-tag' style='background:#dcfce7; color:#166534;'>{s}</span>" for s in match_info["matched_required"]]) + "</div>"
    
    if match_info["matched_preferred"]:
        html += f"<div style='margin-top:8px; margin-bottom:6px;'><span style='color:#92400e;'>✓ Matching Preferred Skills ({len(match_info['matched_preferred'])}):</span></div>"
        html += "<div class='skills'>" + "".join([f"<span class='skill-tag' style='background:#fef3c7; color:#92400e;'>{s}</span>" for s in match_info["matched_preferred"]]) + "</div>"
    
    matched_domain = match_info.get("matched_domain", [])
    if matched_domain:
        html += f"<div style='margin-top:8px; margin-bottom:6px;'><span style='color:#6366f1;'>✓ Matching Domain/Industry ({len(matched_domain)}):</span></div>"
        html += "<div class='skills'>" + "".join([f"<span class='skill-tag' style='background:#e0e7ff; color:#4338ca;'>{d}</span>" for d in matched_domain]) + "</div>"
    
    if match_info["missing_required"]:
        html += f"<div style='margin-top:8px; margin-bottom:6px;'><span style='color:#dc2626;'>✗ Missing Required Skills ({len(match_info['missing_required'])}):</span></div>"
        html += "<div class='skills'>" + "".join([f"<span class='skill-tag' style='background:#fee2e2; color:#dc2626;'>{s}</span>" for s in match_info["missing_required"]]) + "</div>"
    
    if not match_info["matched_required"] and not match_info["matched_preferred"] and not matched_domain:
        html += "<div class='muted'>No skill or domain matches found.</div>"
    
    html += "</div>"
    return html


# =============================================================================
# Google Sheets Sync
# =============================================================================

class GoogleSheetsSync:
    """Fetches job data from a Google Sheet and syncs to local DB."""

    def __init__(self, credentials_file: str, sheet_id: str, sheet_name: str = "Sheet1"):
        self.credentials_file = credentials_file
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name
        self._client = None

    def _get_client(self) -> gspread.Client:
        if self._client is None:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            creds = Credentials.from_service_account_file(self.credentials_file, scopes=scopes)
            self._client = gspread.authorize(creds)
        return self._client

    def is_configured(self) -> bool:
        return bool(self.sheet_id) and os.path.exists(self.credentials_file)

    def fetch_jobs(self) -> Tuple[List[dict], Optional[str]]:
        """Fetch jobs from Google Sheet. Expected columns: Job Title, Job Description, Posted Date.
        Returns (list_of_jobs, error_message)."""
        if not self.is_configured():
            return [], "Google Sheets not configured. Set GOOGLE_SHEET_ID env var and provide google_credentials.json."
        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.worksheet(self.sheet_name)
            records = worksheet.get_all_records()

            jobs = []
            cutoff = datetime.now(timezone.utc) - timedelta(days=JOB_RETENTION_DAYS)
            for row in records:
                title = str(row.get("Job Title", "") or "").strip()
                description = str(row.get("Job Description", "") or "").strip()
                posted_date_str = str(row.get("Posted Date", "") or "").strip()

                if not title or not description:
                    continue

                # Parse posted date (try common formats)
                posted_date = None
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"):
                    try:
                        posted_date = datetime.strptime(posted_date_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue

                if posted_date is None:
                    # If date can't be parsed, default to today
                    posted_date = datetime.now(timezone.utc)

                # Skip jobs older than retention period
                if posted_date < cutoff:
                    continue

                jobs.append({
                    "title": title,
                    "description": description,
                    "posted_date": posted_date.strftime("%Y-%m-%d"),
                })
            return jobs, None
        except gspread.exceptions.SpreadsheetNotFound:
            return [], f"Spreadsheet not found. Make sure the sheet is shared with the service account."
        except Exception as e:
            return [], f"Google Sheets error: {str(e)}"

    def _parse_posted_date(self, posted_date_str: str) -> Optional[datetime]:
        """Parse a posted date string into a datetime object."""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(posted_date_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def cleanup_sheet(self) -> dict:
        """Move expired rows (>1 month) to an archive tab and delete them from the main sheet.
        Returns {"archived": int, "archive_tab": str, "error": str|None}."""
        if not self.is_configured():
            return {"archived": 0, "archive_tab": "", "error": "Google Sheets not configured."}
        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.worksheet(self.sheet_name)

            all_values = worksheet.get_all_values()
            if len(all_values) <= 1:
                return {"archived": 0, "archive_tab": "", "error": None}

            header = all_values[0]
            cutoff = datetime.now(timezone.utc) - timedelta(days=JOB_RETENTION_DAYS)

            # Find Posted Date column index
            try:
                date_col = header.index("Posted Date")
            except ValueError:
                return {"archived": 0, "archive_tab": "", "error": "Column 'Posted Date' not found in sheet."}

            # Separate expired vs active rows
            expired_rows = []
            active_rows = []
            for row in all_values[1:]:
                posted_date_str = row[date_col].strip() if date_col < len(row) else ""
                posted_date = self._parse_posted_date(posted_date_str)
                if posted_date and posted_date < cutoff:
                    expired_rows.append(row)
                else:
                    active_rows.append(row)

            if not expired_rows:
                return {"archived": 0, "archive_tab": "", "error": None}

            # Create or get archive tab named "Archive_YYYY_MM"
            archive_tab_name = f"Archive_{datetime.now(timezone.utc).strftime('%Y_%m')}"
            try:
                archive_ws = spreadsheet.worksheet(archive_tab_name)
            except gspread.exceptions.WorksheetNotFound:
                archive_ws = spreadsheet.add_worksheet(title=archive_tab_name, rows=1, cols=len(header))
                archive_ws.append_row(header, value_input_option="RAW")

            # Append expired rows to archive tab
            archive_ws.append_rows(expired_rows, value_input_option="RAW")

            # Rewrite main sheet with only active rows
            worksheet.clear()
            worksheet.append_row(header, value_input_option="RAW")
            if active_rows:
                worksheet.append_rows(active_rows, value_input_option="RAW")

            return {"archived": len(expired_rows), "archive_tab": archive_tab_name, "error": None}

        except Exception as e:
            return {"archived": 0, "archive_tab": "", "error": f"Sheet cleanup error: {str(e)}"}

    def sync(self, database: 'Database', skill_extractor: 'SkillExtractor',
             embedding_matcher: 'EmbeddingMatcher') -> dict:
        """Sync Google Sheet jobs into the database. Returns summary dict."""
        # Step 1: Cleanup expired jobs from DB
        deleted = database.cleanup_expired_jobs(JOB_RETENTION_DAYS)

        # Step 2: Archive and clean expired rows from Google Sheet
        sheet_cleanup = self.cleanup_sheet()

        # Step 3: Fetch fresh data
        jobs, error = self.fetch_jobs()
        if error:
            return {"error": error, "deleted": deleted, "added": 0, "skipped": 0,
                    "archived": sheet_cleanup.get("archived", 0),
                    "archive_tab": sheet_cleanup.get("archive_tab", ""),
                    "sheet_cleanup_error": sheet_cleanup.get("error")}

        added = 0
        skipped = 0
        for job in jobs:
            # Skip if already exists
            if database.job_exists_by_title_and_source(job["title"], "google_sheet"):
                skipped += 1
                continue

            # Extract skills via LLM
            job_data, llm_err = skill_extractor.extract_from_job(job["title"], job["description"])
            if llm_err:
                job_data = JobData()
                logger.warning("LLM extraction failed for '%s': %s", job["title"], llm_err)

            # Create embedding and store
            embedding = embedding_matcher.embed_job(job_data, job["title"], job["description"])
            database.insert_job(
                title=job["title"],
                description=job["description"],
                extracted_data=job_data,
                embedding=embedding,
                posted_date=job["posted_date"],
                source="google_sheet",
            )

            # Track skills
            database.track_skills({
                "technical": job_data.required_technical_skills + job_data.preferred_technical_skills,
                "business": job_data.required_business_skills + job_data.preferred_business_skills,
                "domain": job_data.required_domain_expertise,
                "soft": job_data.required_soft_skills,
                "certifications": job_data.required_certifications + job_data.preferred_certifications,
            }, source="job")
            added += 1

        return {"error": None, "deleted": deleted, "added": added, "skipped": skipped, "total_fetched": len(jobs),
                "archived": sheet_cleanup.get("archived", 0),
                "archive_tab": sheet_cleanup.get("archive_tab", ""),
                "sheet_cleanup_error": sheet_cleanup.get("error")}


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title=APP_TITLE)

# Global instances (initialized on startup)
db: Optional[Database] = None
llm_client: Optional[LLMClient] = None
extractor: Optional[SkillExtractor] = None
matcher: Optional[EmbeddingMatcher] = None
vstore: Optional[VectorStore] = None
gsheet_sync: Optional[GoogleSheetsSync] = None
gdrive_client: Optional[GoogleDriveClient] = None
supabase_client: Optional[SupabaseClient] = None


@app.get("/resume/{filename}")
def serve_resume(filename: str):
    """Serve resume file for viewing/downloading."""
    # Check storage type
    resume = db.get_resume_by_filename(filename)
    if resume:
        storage_type = resume.get("storage_type", "local")
        file_path = resume.get("file_path")

        # Supabase Storage
        if storage_type == "supabase" and supabase_client and supabase_client.is_configured():
            signed_url = supabase_client.get_download_url(file_path)
            if signed_url:
                return RedirectResponse(signed_url)

        # Google Drive
        elif storage_type == "gdrive" and gdrive_client and gdrive_client.is_configured():
            drive_url = gdrive_client.get_download_url(file_path)
            return RedirectResponse(drive_url)

    # Fall back to local file
    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    file_path = os.path.join(uploads_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return HTMLResponse("<h1>File not found</h1>", status_code=404)


@app.on_event("startup")
def on_startup() -> None:
    global db, llm_client, extractor, matcher, vstore, gsheet_sync, gdrive_client, supabase_client

    # Initialize vector store and database based on configuration
    if USE_SUPABASE_DB:
        if SUPABASE_DB_URL:
            vstore = SupabaseVectorStore(SUPABASE_DB_URL)
            db = SupabaseDatabase(SUPABASE_DB_URL, vector_store=vstore)
            logger.info("Supabase PostgreSQL database enabled")
        else:
            logger.warning("USE_SUPABASE_DB=true but SUPABASE_DB_URL not set, falling back to SQLite")
            if ENABLE_CHROMA:
                vstore = VectorStore(CHROMA_DB_PATH)
                logger.info("ChromaDB enabled at: %s", CHROMA_DB_PATH)
            else:
                vstore = None
                logger.info("ChromaDB disabled for memory optimization")
            db = Database(vector_store=vstore)
    else:
        # Initialize ChromaDB vector store (optional for memory-constrained deployments)
        if ENABLE_CHROMA:
            vstore = VectorStore(CHROMA_DB_PATH)
            logger.info("ChromaDB enabled at: %s", CHROMA_DB_PATH)
        else:
            vstore = None
            logger.info("ChromaDB disabled for memory optimization")
        # Initialize database with vector store (optional)
        db = Database(vector_store=vstore)
    # Initialize LLM client based on LLM_PROVIDER env var
    llm_client = create_llm_client()
    logger.info("LLM provider: %s", llm_client.get_provider_name())
    extractor = SkillExtractor(llm_client)
    matcher = EmbeddingMatcher()  # Uses OpenAI embeddings to save memory
    gsheet_sync = GoogleSheetsSync(GOOGLE_SHEETS_CREDENTIALS_FILE, GOOGLE_SHEET_ID, GOOGLE_SHEET_NAME)
    # Initialize Google Drive client if enabled
    if USE_GOOGLE_DRIVE:
        gdrive_client = GoogleDriveClient()
        logger.info("Google Drive storage enabled")
    else:
        gdrive_client = None
        logger.info("Google Drive storage disabled")

    # Initialize Supabase client if enabled
    if USE_SUPABASE:
        supabase_client = SupabaseClient()
        if supabase_client.is_configured():
            logger.info("Supabase storage enabled")
        else:
            supabase_client = None
            logger.info("Supabase storage disabled (credentials not configured)")
    else:
        supabase_client = None
        logger.info("Supabase storage disabled")

    # Migrate existing embeddings from SQLite JSON blobs into ChromaDB (one-time)
    # Skip migration if using Supabase
    if ENABLE_CHROMA and not USE_SUPABASE_DB:
        _migrate_embeddings_to_chroma(db, vstore)

    # Auto-sync Google Sheet jobs on startup (cleanup expired + fetch new)
    if gsheet_sync.is_configured():
        try:
            result = gsheet_sync.sync(db, extractor, matcher)
            if result.get("error"):
                logger.warning("Google Sheets sync error on startup: %s", result["error"])
            else:
                logger.info("Google Sheets sync on startup: added=%d, skipped=%d, deleted=%d",
                            result["added"], result["skipped"], result["deleted"])
        except Exception as e:
            logger.warning("Google Sheets sync failed on startup: %s", str(e))

    # Force garbage collection to free memory after startup
    gc.collect()
    logger.info("Startup complete, memory freed via garbage collection")


def _migrate_embeddings_to_chroma(database: Database, vector_store: VectorStore) -> None:
    """One-time migration: copy embeddings from SQLite JSON blobs into ChromaDB."""
    if vector_store.jobs.count() == 0:
        jobs = database.get_all_jobs()
        migrated = 0
        for j in jobs:
            try:
                emb = EmbeddingMatcher.embedding_from_json(j["embedding_json"])
                vector_store.add_job(j["id"], emb)
                migrated += 1
            except Exception:
                pass
        if migrated:
            logger.info("Migrated %d job embeddings to ChromaDB", migrated)

    if vector_store.resumes.count() == 0:
        resumes = database.get_all_resumes()
        migrated = 0
        for r in resumes:
            try:
                emb = EmbeddingMatcher.embedding_from_json(r["embedding_json"])
                vector_store.add_resume(r["id"], emb)
                migrated += 1
            except Exception:
                pass
        if migrated:
            logger.info("Migrated %d resume embeddings to ChromaDB", migrated)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    llm_status = "online" if llm_client and llm_client.is_available() else "offline"
    provider_name = llm_client.get_provider_name() if llm_client else "Not configured"
    status_text = f"{provider_name}: <span class='status {llm_status}'>{llm_status}</span>"

    body = f"""
    <h1>Resume - Job Matcher</h1>
    <p class='muted'>Universal skill extraction for ALL industries: Tech, Finance, Healthcare, MBA, Legal, Marketing, and more</p>
    <div class='card'>
      <div class='card-header'>System Status</div>
      <div>{status_text}</div>
      <div style='margin-top:12px'>
        <a href='/sync_jobs'><button style='background:#059669;'>Sync Jobs from Google Sheet</button></a>
      </div>
    </div>
    <div class='card'>
      <div class='card-header'>Jobs Board</div>
      <div class='muted'>Browse all jobs (including auto-synced from Google Sheets) - Search jobs - Find relevant jobs for a candidate</div>
      <div style='margin-top:12px'><a href='/jobs'><button>Go to Jobs Board</button></a></div>
    </div>
    <div class='card'>
      <div class='card-header'>Candidate Portal</div>
      <div class='muted'>Upload your resume (PDF/DOCX) - LLM extracts your skills across all categories - See matching jobs</div>
      <div style='margin-top:12px'><a href='/candidate'><button>Go to Candidate Portal</button></a></div>
    </div>
    <div class='card'>
      <div class='card-header'>Recruiter Portal</div>
      <div class='muted'>Post any job (Software, Finance, Marketing, etc.) - LLM extracts requirements - See matching candidates</div>
      <div style='margin-top:12px'><a href='/recruiter'><button>Go to Recruiter Portal</button></a></div>
    </div>
    <div class='card'>
      <div class='card-header'>Supported Industries</div>
      <div class='skills'>
        <span class='skill-tag technical'>Software & IT</span>
        <span class='skill-tag business'>MBA & Management</span>
        <span class='skill-tag domain'>Finance & Banking</span>
        <span class='skill-tag soft'>Healthcare</span>
        <span class='skill-tag cert'>Legal</span>
        <span class='skill-tag tool'>Marketing & Sales</span>
        <span class='skill-tag'>HR & Operations</span>
        <span class='skill-tag'>Education</span>
        <span class='skill-tag'>Consulting</span>
        <span class='skill-tag'>And more...</span>
      </div>
    </div>
    """
    return html_page(APP_TITLE, body)


@app.get("/candidate", response_class=HTMLResponse)
def candidate_portal() -> str:
    llm_available = llm_client and llm_client.is_available()
    provider_name = llm_client.get_provider_name() if llm_client else "Not configured"
    warning = "" if llm_available else f"<div class='warning'>LLM ({provider_name}) is not available. Check your LLM_PROVIDER and LLM_API_KEY settings.</div>"
    
    # Get existing resumes for selection
    existing_resumes = db.get_all_resumes() if db else []
    resume_options = ""
    if existing_resumes:
        for r in existing_resumes:
            resume_options += f"<option value='{r['id']}'>{r['filename']}</option>"

    body = f"""
    <h1>Candidate Portal</h1>
    <p class='muted'>Works for all professions: Engineers, MBAs, Doctors, Accountants, Lawyers, Marketers, etc.</p>
    {warning}
    <div class='card'>
      <form action='/upload_resume' method='post' enctype='multipart/form-data'>
        <div class='card-header'>Upload New Resume</div>
        <div class='muted'>Supported formats: PDF, DOCX</div>
        <div style='margin-top:12px'><input type='file' name='resume' accept='.pdf,.docx' required /></div>
        <div style='margin-top:12px'>
          <label><input type='checkbox' name='find_jobs' value='1' checked /> Find matching jobs immediately</label>
        </div>
        <div style='margin-top:16px'><button type='submit'>Upload Resume</button></div>
      </form>
    </div>
    <div class='card'>
      <form action='/upload_resumes' method='post' enctype='multipart/form-data'>
        <div class='card-header'>Bulk Upload Resumes</div>
        <div class='muted'>Upload multiple resumes at once (PDF, DOCX). No matching — just stores and extracts skills.</div>
        <div style='margin-top:12px'><input type='file' name='resumes' accept='.pdf,.docx' multiple required /></div>
        <div style='margin-top:16px'><button type='submit'>Upload All</button></div>
      </form>
    </div>
    {"<div class='card'><form action='/find_jobs_for_resume' method='get'><div class='card-header'>Or Select Existing Resume</div><div class='muted'>Choose from previously uploaded resumes</div><div style='margin-top:12px'><select name='resume_id' required style='padding:8px; width:100%; max-width:400px;'><option value=''>-- Select Resume --</option>" + resume_options + "</select></div><div style='margin-top:16px'><button type='submit'>Find Matching Jobs</button></div></form></div>" if existing_resumes else ""}
    """
    return html_page("Candidate Portal", body)


@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(resume: UploadFile = File(...), find_jobs: str = Form(None)) -> str:
    if not all([db, extractor, matcher]):
        return html_page("Error", "", error="System not initialized")

    # Save uploaded file
    safe_name = (resume.filename or "resume").replace("/", "_").replace("\\", "_")
    content_type = resume.content_type or "application/octet-stream"
    storage_type = "local"

    if supabase_client and supabase_client.is_configured():
        # Upload to Supabase Storage
        content = await resume.read()
        storage_path = supabase_client.upload_file(content, safe_name, content_type)
        if not storage_path:
            return html_page("Candidate Portal", f"<a href='/candidate'>Back</a>", error="Failed to upload to Supabase Storage")
        file_path = storage_path  # Store Supabase storage path
        storage_type = "supabase"
    elif gdrive_client and gdrive_client.is_configured():
        # Upload to Google Drive
        content = await resume.read()
        drive_file_id = gdrive_client.upload_file(content, safe_name, content_type)
        if not drive_file_id:
            return html_page("Candidate Portal", f"<a href='/candidate'>Back</a>", error="Failed to upload to Google Drive")
        file_path = drive_file_id  # Store Drive file ID instead of local path
        storage_type = "gdrive"
    else:
        # Save to local disk
        uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, safe_name)
        # Read file and save to disk
        content = await resume.read()
        with open(file_path, "wb") as f:
            f.write(content)

    # Extract text (for Supabase/Google Drive, download first)
    if storage_type == "supabase":
        content = supabase_client.get_file(file_path)
        if content:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            raw_text, parse_error = DocumentParser.extract_text(tmp_path, safe_name)
            os.unlink(tmp_path)
        else:
            raw_text = ""
            parse_error = "Failed to download from Supabase Storage"
    elif storage_type == "gdrive":
        content = gdrive_client.get_file(file_path)
        if content:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            raw_text, parse_error = DocumentParser.extract_text(tmp_path, safe_name)
            os.unlink(tmp_path)
        else:
            raw_text = ""
            parse_error = "Failed to download from Google Drive"
    else:
        raw_text, parse_error = DocumentParser.extract_text(file_path, safe_name)
    if parse_error:
        return html_page("Candidate Portal", f"<a href='/candidate'>Back</a>", error=parse_error)

    # Extract skills via LLM
    resume_data, llm_error = extractor.extract_from_resume(raw_text)
    if llm_error:
        resume_data = ResumeData()
        llm_warning = f"<div class='warning'>LLM extraction failed: {llm_error}. Using skill-based matching only.</div>"
    else:
        llm_warning = ""

    # Create embedding and store (will override if same filename exists)
    embedding = matcher.embed_resume(resume_data, raw_text)
    db.insert_resume(safe_name, resume.content_type, raw_text, resume_data, embedding, file_path, storage_type)
    
    # Track skills in taxonomy for learning
    db.track_skills({
        "technical": resume_data.technical_skills,
        "business": resume_data.business_skills,
        "domain": resume_data.domain_expertise,
        "soft": resume_data.soft_skills,
        "tools": resume_data.tools_and_software,
        "certifications": resume_data.certifications,
    }, source="resume")

    # If user unchecked "find jobs", just show upload success
    if not find_jobs:
        skills_html = ""
        skills_html += render_skills_section("Technical Skills", resume_data.technical_skills, "technical")
        skills_html += render_skills_section("Business Skills", resume_data.business_skills, "business")
        skills_html += render_skills_section("Domain Expertise", resume_data.domain_expertise, "domain")
        skills_html += render_skills_section("Soft Skills", resume_data.soft_skills, "soft")
        skills_html += render_skills_section("Tools & Software", resume_data.tools_and_software, "tool")
        skills_html += render_skills_section("Certifications", resume_data.certifications, "cert")
        
        body = f"""
        <h1>Resume Uploaded Successfully</h1>
        {llm_warning}
        <div class='card'>
          <div class='card-header'>{safe_name}</div>
          {skills_html}
          {f"<div class='summary'>{resume_data.summary}</div>" if resume_data.summary else ""}
        </div>
        <div style='margin-top:20px'>
          <a href='/find_jobs_for_resume?resume_id={safe_name}'><button>Find Matching Jobs</button></a>
          <a href='/candidate' style='margin-left:10px;'>Upload Another Resume</a>
        </div>
        """
        return html_page("Resume Uploaded", body)

    # Find matching jobs using blended semantic + skill matching
    jobs = db.get_all_jobs()
    scored_jobs = rank_by_skills(resume_data, jobs,
                                vector_store=vstore,
                                resume_embedding=embedding)
    # Filter: only show jobs with at least 2 matched skills
    scored_jobs = [(score, job, match_info) for score, job, match_info in scored_jobs
                   if len(match_info["matched_required"]) + len(match_info["matched_preferred"]) + len(match_info["matched_domain"]) >= 2][:10]

    # Render extracted skills
    skills_html = ""
    skills_html += render_skills_section("Technical Skills", resume_data.technical_skills, "technical")
    skills_html += render_skills_section("Business Skills", resume_data.business_skills, "business")
    skills_html += render_skills_section("Domain Expertise", resume_data.domain_expertise, "domain")
    skills_html += render_skills_section("Soft Skills", resume_data.soft_skills, "soft")
    skills_html += render_skills_section("Tools & Software", resume_data.tools_and_software, "tool")
    skills_html += render_skills_section("Certifications", resume_data.certifications, "cert")

    # Render matching jobs
    jobs_html = ""
    if not scored_jobs:
        jobs_html = "<div class='card muted'>No jobs in database yet. Ask a recruiter to add some!</div>"
    else:
        for score, job, match_info in scored_jobs:
            job_data = JobData.from_json(job["extracted_data_json"])
            industry_badge = f"<span class='badge industry'>{job_data.industry}</span>" if job_data.industry else ""
            seniority_badge = f"<span class='badge seniority'>{job_data.seniority_level}</span>" if job_data.seniority_level else ""
            # Job meta info
            job_meta_parts = []
            if job_data.department:
                job_meta_parts.append(f"<b>Department:</b> {job_data.department}")
            if job_data.job_type:
                job_meta_parts.append(f"<b>Type:</b> {job_data.job_type}")
            if job_data.experience_years_min or job_data.experience_years_max:
                exp = f"{job_data.experience_years_min or 0}-{job_data.experience_years_max or '?'} years"
                job_meta_parts.append(f"<b>Experience:</b> {exp}")
            if job_data.required_education:
                job_meta_parts.append(f"<b>Education:</b> {', '.join(job_data.required_education[:2])}")
            job_meta_html = " | ".join(job_meta_parts) if job_meta_parts else ""
            # Job description snippet
            job_desc_snippet = snippet(job["description"], 500)
            # Use match_info from ranking (already computed)
            match_analysis_html = render_match_analysis(match_info)
            jobs_html += f"""
            <div class='card'>
              <div class='card-header'>{job["title"]}{industry_badge}{seniority_badge}</div>
              <div class='score'>Match: {score:.1%} <span class='muted' style='font-weight:normal;'>(Skill: {match_info['match_score']:.0%} | Semantic: {match_info['semantic_score']:.0%})</span></div>
              {f"<div class='meta'>{job_meta_html}</div>" if job_meta_html else ""}
              {match_analysis_html}
              <div style='margin-top:10px;'><b>Job Description:</b></div>
              <pre class='muted' style='font-family:inherit; margin-top:4px;'>{job_desc_snippet}</pre>
              {render_skills_section("Required Technical", job_data.required_technical_skills, "technical")}
              {render_skills_section("Required Business", job_data.required_business_skills, "business")}
              {render_skills_section("Required Domain", job_data.required_domain_expertise, "domain")}
              {render_skills_section("Required Soft Skills", job_data.required_soft_skills, "soft")}
              {render_skills_section("Required Certifications", job_data.required_certifications, "cert")}
              {render_skills_section("Preferred Skills", job_data.all_preferred_skills(), "preferred")}
              {f"<div class='summary'><b>Summary:</b> {job_data.summary}</div>" if job_data.summary else ""}
            </div>
            """

    # Meta info
    meta_parts = []
    if resume_data.experience_years:
        meta_parts.append(f"<b>Experience:</b> {resume_data.experience_years} years")
    if resume_data.industries:
        meta_parts.append(f"<b>Industries:</b> {', '.join(resume_data.industries)}")
    if resume_data.education:
        meta_parts.append(f"<b>Education:</b> {', '.join(resume_data.education[:2])}")
    meta_html = " | ".join(meta_parts) if meta_parts else ""

    body = f"""
    <h1>Matching Jobs for Your Resume</h1>
    {llm_warning}
    <div class='card'>
      <div class='card-header'>{safe_name}</div>
      {f"<div class='meta'>{meta_html}</div>" if meta_html else ""}
      {skills_html}
      {f"<div class='summary'>{resume_data.summary}</div>" if resume_data.summary else ""}
    </div>
    <h2>Top Matching Jobs</h2>
    {jobs_html}
    <div style='margin-top:20px'><a href='/candidate'>Upload Another Resume</a></div>
    """
    return html_page("Matching Jobs", body)


@app.post("/upload_resumes", response_class=HTMLResponse)
async def upload_resumes(resumes: List[UploadFile] = File(...)) -> str:
    """Bulk upload multiple resumes — no matching, just store and extract skills."""
    if not all([db, extractor, matcher]):
        return html_page("Error", "", error="System not initialized")

    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    results = []
    for resume in resumes:
        safe_name = (resume.filename or "resume").replace("/", "_").replace("\\", "_")
        content_type = resume.content_type or "application/octet-stream"
        storage_type = "local"

        if supabase_client and supabase_client.is_configured():
            # Upload to Supabase Storage
            content = await resume.read()
            storage_path = supabase_client.upload_file(content, safe_name, content_type)
            if not storage_path:
                results.append({"filename": safe_name, "status": "error", "error": "Failed to upload to Supabase Storage"})
                continue
            file_path = storage_path
            storage_type = "supabase"
        elif gdrive_client and gdrive_client.is_configured():
            # Upload to Google Drive
            content = await resume.read()
            drive_file_id = gdrive_client.upload_file(content, safe_name, content_type)
            if not drive_file_id:
                results.append({"filename": safe_name, "status": "error", "error": "Failed to upload to Google Drive"})
                continue
            file_path = drive_file_id
            storage_type = "gdrive"
        else:
            # Save to local disk
            file_path = os.path.join(uploads_dir, safe_name)
            # Read file and save to disk
            content = await resume.read()
            with open(file_path, "wb") as f:
                f.write(content)

        # Extract text (for Supabase/Google Drive, download first)
        if storage_type == "supabase":
            content = supabase_client.get_file(file_path)
            if content:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                raw_text, parse_error = DocumentParser.extract_text(tmp_path, safe_name)
                os.unlink(tmp_path)
            else:
                raw_text = ""
                parse_error = "Failed to download from Supabase Storage"
        elif storage_type == "gdrive":
            content = gdrive_client.get_file(file_path)
            if content:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_name)[1]) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                raw_text, parse_error = DocumentParser.extract_text(tmp_path, safe_name)
                os.unlink(tmp_path)
            else:
                raw_text = ""
                parse_error = "Failed to download from Google Drive"
        else:
            raw_text, parse_error = DocumentParser.extract_text(file_path, safe_name)
        if parse_error:
            results.append({"filename": safe_name, "status": "error", "error": parse_error})
            continue

        # Extract skills via LLM
        resume_data, llm_error = extractor.extract_from_resume(raw_text)
        if llm_error:
            resume_data = ResumeData()

        # Create embedding and store
        embedding = matcher.embed_resume(resume_data, raw_text)
        db.insert_resume(safe_name, resume.content_type, raw_text, resume_data, embedding, file_path, storage_type)

        # Track skills
        db.track_skills({
            "technical": resume_data.technical_skills,
            "business": resume_data.business_skills,
            "domain": resume_data.domain_expertise,
            "soft": resume_data.soft_skills,
            "tools": resume_data.tools_and_software,
            "certifications": resume_data.certifications,
        }, source="resume")

        results.append({
            "filename": safe_name,
            "status": "success",
            "skills_count": len(resume_data.all_skills()),
            "llm_warning": llm_error,
        })

    # Render results
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    cards_html = ""
    for r in results:
        if r["status"] == "success":
            warning = f"<div class='warning' style='margin-top:6px; padding:6px 10px; font-size:0.85em;'>LLM warning: {r['llm_warning']}</div>" if r.get("llm_warning") else ""
            cards_html += f"""
            <div class='card'>
              <div class='card-header' style='color:#166534;'>&#10003; {r['filename']}</div>
              <div class='muted'>{r['skills_count']} skills extracted</div>
              {warning}
            </div>"""
        else:
            cards_html += f"""
            <div class='card'>
              <div class='card-header' style='color:#dc2626;'>&#10007; {r['filename']}</div>
              <div class='error' style='margin-top:4px; padding:6px 10px; font-size:0.85em;'>{r['error']}</div>
            </div>"""

    body = f"""
    <h1>Bulk Upload Complete</h1>
    <div class='{"success" if error_count == 0 else "warning"}'>{success_count} uploaded successfully, {error_count} failed</div>
    {cards_html}
    <div style='margin-top:20px'>
      <a href='/candidate'><button>Back to Candidate Portal</button></a>
      <a href='/jobs' style='margin-left:10px;'><button style='background:#059669;'>Find Jobs</button></a>
    </div>
    """
    return html_page("Bulk Upload Results", body)


@app.get("/find_jobs_for_resume", response_class=HTMLResponse)
def find_jobs_for_resume(resume_id: str) -> str:
    """Find matching jobs for an existing resume."""
    if not db:
        return html_page("Error", "", error="Database not initialized")
    
    # Get resume by ID or filename
    resumes = db.get_all_resumes()
    resume = None
    for r in resumes:
        if str(r["id"]) == resume_id or r["filename"] == resume_id:
            resume = r
            break
    
    if not resume:
        return html_page("Error", "<a href='/candidate'>Back</a>", error="Resume not found")
    
    resume_data = ResumeData.from_json(resume["extracted_data_json"])
    resume_emb = vstore.get_resume_embedding(resume["id"]) if vstore else None
    
    # Find matching jobs using blended semantic + skill matching
    jobs = db.get_all_jobs()
    scored_jobs = rank_by_skills(resume_data, jobs,
                                vector_store=vstore,
                                resume_embedding=resume_emb)
    # Filter: only show jobs with at least 2 matched skills
    scored_jobs = [(score, job, match_info) for score, job, match_info in scored_jobs
                   if len(match_info["matched_required"]) + len(match_info["matched_preferred"]) + len(match_info["matched_domain"]) >= 2][:10]
    
    # Render extracted skills
    skills_html = ""
    skills_html += render_skills_section("Technical Skills", resume_data.technical_skills, "technical")
    skills_html += render_skills_section("Business Skills", resume_data.business_skills, "business")
    skills_html += render_skills_section("Domain Expertise", resume_data.domain_expertise, "domain")
    skills_html += render_skills_section("Soft Skills", resume_data.soft_skills, "soft")
    skills_html += render_skills_section("Tools & Software", resume_data.tools_and_software, "tool")
    skills_html += render_skills_section("Certifications", resume_data.certifications, "cert")
    
    # Render matching jobs
    jobs_html = ""
    if not scored_jobs:
        jobs_html = "<div class='card muted'>No jobs in database yet. Ask a recruiter to add some!</div>"
    else:
        for score, job, match_info in scored_jobs:
            job_data = JobData.from_json(job["extracted_data_json"])
            industry_badge = f"<span class='badge industry'>{job_data.industry}</span>" if job_data.industry else ""
            seniority_badge = f"<span class='badge seniority'>{job_data.seniority_level}</span>" if job_data.seniority_level else ""
            job_meta_parts = []
            if job_data.department:
                job_meta_parts.append(f"<b>Department:</b> {job_data.department}")
            if job_data.job_type:
                job_meta_parts.append(f"<b>Type:</b> {job_data.job_type}")
            if job_data.experience_years_min or job_data.experience_years_max:
                exp = f"{job_data.experience_years_min or 0}-{job_data.experience_years_max or '?'} years"
                job_meta_parts.append(f"<b>Experience:</b> {exp}")
            job_meta_html = " | ".join(job_meta_parts) if job_meta_parts else ""
            job_desc_snippet = snippet(job["description"], 500)
            match_analysis_html = render_match_analysis(match_info)
            jobs_html += f"""
            <div class='card'>
              <div class='card-header'>{job["title"]}{industry_badge}{seniority_badge}</div>
              <div class='score'>Match: {score:.1%} <span class='muted' style='font-weight:normal;'>(Skill: {match_info['match_score']:.0%} | Semantic: {match_info['semantic_score']:.0%})</span></div>
              {f"<div class='meta'>{job_meta_html}</div>" if job_meta_html else ""}
              {match_analysis_html}
              <div style='margin-top:10px;'><b>Job Description:</b></div>
              <pre class='muted' style='font-family:inherit; margin-top:4px;'>{job_desc_snippet}</pre>
            </div>
            """
    
    body = f"""
    <h1>Matching Jobs for {resume["filename"]}</h1>
    <div class='card'>
      <div class='card-header'>{resume["filename"]} <a href='/resume/{resume["filename"]}' target='_blank' style='font-size:0.8em; margin-left:10px;'>View Resume</a></div>
      {skills_html}
      {f"<div class='summary'>{resume_data.summary}</div>" if resume_data.summary else ""}
    </div>
    <h2>Top Matching Jobs</h2>
    {jobs_html}
    <div style='margin-top:20px'><a href='/candidate'>Back to Candidate Portal</a></div>
    """
    return html_page("Matching Jobs", body)


@app.get("/recruiter", response_class=HTMLResponse)
def recruiter_portal() -> str:
    llm_available = llm_client and llm_client.is_available()
    provider_name = llm_client.get_provider_name() if llm_client else "Not configured"
    warning = "" if llm_available else f"<div class='warning'>LLM ({provider_name}) is not available. Check your LLM_PROVIDER and LLM_API_KEY settings.</div>"

    body = f"""
    <h1>Recruiter Portal</h1>
    <p class='muted'>Post jobs from any industry: Software, Finance, Healthcare, Marketing, Legal, etc.</p>
    {warning}
    <div class='card'>
      <form action='/create_job' method='post'>
        <div class='card-header'>Post a Job Opening</div>
        <div style='margin-top:12px'>
          <label class='muted'><b>Job Title</b></label>
          <input type='text' name='title' placeholder='e.g. Senior Software Engineer, Investment Analyst, Marketing Manager, Registered Nurse' required />
        </div>
        <div style='margin-top:16px'>
          <label class='muted'><b>Job Description</b></label>
          <textarea name='description' placeholder='Paste the full job description here including requirements, responsibilities, qualifications, required skills, preferred skills, experience requirements...' required></textarea>
        </div>
        <div style='margin-top:16px'>
          <label class='muted'><b>Posted By</b></label>
          <input type='text' name='posted_by' placeholder='e.g. Your Name or Company' />
        </div>
        <div style='margin-top:16px'><button type='submit'>Extract Skills & Find Candidates</button></div>
      </form>
    </div>
    """
    return html_page("Recruiter Portal", body)


@app.post("/create_job", response_class=HTMLResponse)
def create_job(title: str = Form(...), description: str = Form(...), posted_by: str = Form("")) -> str:
    if not all([db, extractor, matcher]):
        return html_page("Error", "", error="System not initialized")

    title = (title or "").strip()
    description = (description or "").strip()
    if not title or not description:
        return html_page("Recruiter Portal", "", error="Title and description are required")

    # Extract skills via LLM
    job_data, llm_error = extractor.extract_from_job(title, description)
    if llm_error:
        job_data = JobData()
        llm_warning = f"<div class='warning'>LLM extraction failed: {llm_error}. Using embedding-only matching.</div>"
    else:
        llm_warning = ""

    # Create embedding and store
    embedding = matcher.embed_job(job_data, title, description)
    posted_by = (posted_by or "").strip()
    db.insert_job(title, description, job_data, embedding, posted_by=posted_by)
    
    # Track skills in taxonomy for learning
    db.track_skills({
        "technical": job_data.required_technical_skills + job_data.preferred_technical_skills,
        "business": job_data.required_business_skills + job_data.preferred_business_skills,
        "domain": job_data.required_domain_expertise,
        "soft": job_data.required_soft_skills,
        "certifications": job_data.required_certifications + job_data.preferred_certifications,
    }, source="job")

    # Find matching resumes using blended semantic + skill matching
    resumes = db.get_all_resumes()
    scored_resumes = rank_resumes_by_skills(job_data, resumes, title,
                                            vector_store=vstore,
                                            job_embedding=embedding)
    # Filter: only show resumes with at least 2 matched skills
    scored_resumes = [(score, resume, match_info) for score, resume, match_info in scored_resumes
                      if len(match_info["matched_required"]) + len(match_info["matched_preferred"]) + len(match_info["matched_domain"]) >= 2][:10]

    # Render extracted requirements
    req_html = ""
    req_html += render_skills_section("Required Technical", job_data.required_technical_skills, "technical")
    req_html += render_skills_section("Required Business", job_data.required_business_skills, "business")
    req_html += render_skills_section("Required Domain", job_data.required_domain_expertise, "domain")
    req_html += render_skills_section("Required Soft Skills", job_data.required_soft_skills, "soft")
    req_html += render_skills_section("Required Certifications", job_data.required_certifications, "cert")
    req_html += render_skills_section("Preferred Skills", job_data.all_preferred_skills(), "preferred")

    # Render matching resumes
    resumes_html = ""
    if not scored_resumes:
        resumes_html = "<div class='card muted'>No resumes in database yet. Wait for candidates to upload!</div>"
    else:
        for score, res, match_info in scored_resumes:
            res_data = ResumeData.from_json(res["extracted_data_json"])
            # Use match_info from ranking (already computed)
            match_analysis_html = render_match_analysis(match_info)
            resumes_html += f"""
            <div class='card'>
              <div class='card-header'>{res["filename"]} <a href='/resume/{res["filename"]}' target='_blank' style='font-size:0.8em; margin-left:10px;'>View Resume</a></div>
              <div class='score'>Match: {score:.1%} <span class='muted' style='font-weight:normal;'>(Skill: {match_info['match_score']:.0%} | Semantic: {match_info['semantic_score']:.0%})</span></div>
              {f"<div><b>Name:</b> {res_data.name}</div>" if res_data.name else ""}
              <div class='meta'>
                {f"<span><b>Experience:</b> {res_data.experience_years} years</span>" if res_data.experience_years else ""}
                {f"<span><b>Industries:</b> {', '.join(res_data.industries)}</span>" if res_data.industries else ""}
                {f"<span><b>Education:</b> {', '.join(res_data.education[:2])}</span>" if res_data.education else ""}
              </div>
              {match_analysis_html}
              {render_skills_section("Technical Skills", res_data.technical_skills, "technical")}
              {render_skills_section("Business Skills", res_data.business_skills, "business")}
              {render_skills_section("Domain Expertise", res_data.domain_expertise, "domain")}
              {render_skills_section("Soft Skills", res_data.soft_skills, "soft")}
              {render_skills_section("Tools & Software", res_data.tools_and_software, "tool")}
              {render_skills_section("Certifications", res_data.certifications, "cert")}
              {f"<div class='summary'>{res_data.summary}</div>" if res_data.summary else ""}
            </div>
            """

    # Meta info
    meta_parts = []
    if job_data.industry:
        meta_parts.append(f"<b>Industry:</b> {job_data.industry}")
    if job_data.department:
        meta_parts.append(f"<b>Department:</b> {job_data.department}")
    if job_data.seniority_level:
        meta_parts.append(f"<b>Seniority:</b> {job_data.seniority_level}")
    if job_data.experience_years_min or job_data.experience_years_max:
        exp = f"{job_data.experience_years_min or 0}-{job_data.experience_years_max or '?'} years"
        meta_parts.append(f"<b>Experience:</b> {exp}")
    meta_html = " | ".join(meta_parts) if meta_parts else ""

    body = f"""
    <h1>Matching Candidates for Your Job</h1>
    {llm_warning}
    <div class='card'>
      <div class='card-header'>{title}</div>
      {f"<div class='meta'>Posted by: <b>{posted_by}</b> | {meta_html}</div>" if posted_by else (f"<div class='meta'>{meta_html}</div>" if meta_html else "")}
      {req_html}
      {f"<div class='summary'>{job_data.summary}</div>" if job_data.summary else ""}
    </div>
    <h2>Top Matching Candidates</h2>
    {resumes_html}
    <div style='margin-top:20px'><a href='/recruiter'>Post Another Job</a></div>
    """
    return html_page("Matching Candidates", body)


@app.get("/jobs", response_class=HTMLResponse)
def jobs_board(q: str = Query("", alias="q"), candidate_id: str = Query("", alias="candidate_id")) -> str:
    """Jobs Board: browse all jobs, search, and find relevant jobs for a candidate."""
    if not db:
        return html_page("Jobs Board", "", error="Database not initialized")

    # Get all resumes for candidate dropdown
    all_resumes = db.get_all_resumes()

    # Determine which jobs to show
    search_query = (q or "").strip()
    if search_query:
        jobs = db.search_jobs(search_query)
    else:
        jobs = db.get_all_jobs()

    # If a candidate is selected, rank jobs by relevance
    selected_resume = None
    scored_jobs = None
    if candidate_id:
        for r in all_resumes:
            if str(r["id"]) == candidate_id:
                selected_resume = r
                break
        if selected_resume:
            resume_data = ResumeData.from_json(selected_resume["extracted_data_json"])
            resume_emb = vstore.get_resume_embedding(selected_resume["id"]) if vstore else None
            scored_jobs = rank_by_skills(resume_data, jobs,
                                         vector_store=vstore,
                                         resume_embedding=resume_emb)
            # Filter: only show jobs with at least 2 matched skills
            scored_jobs = [(score, job, match_info) for score, job, match_info in scored_jobs
                           if len(match_info["matched_required"]) + len(match_info["matched_preferred"]) + len(match_info["matched_domain"]) >= 2]

    # Build candidate dropdown options
    candidate_options = ""
    for r in all_resumes:
        selected = "selected" if candidate_id and str(r["id"]) == candidate_id else ""
        candidate_options += f"<option value='{r['id']}' {selected}>{r['filename']}</option>"

    # Google Sheets status
    gsheet_status = ""
    if gsheet_sync and gsheet_sync.is_configured():
        gsheet_status = "<span class='status online'>Google Sheets Connected</span>"
    else:
        gsheet_status = "<span class='status offline'>Google Sheets Not Configured</span>"

    # Build search + filter bar
    controls_html = f"""
    <div class='card'>
      <div style='display:flex; gap:16px; flex-wrap:wrap; align-items:end;'>
        <div style='flex:1; min-width:250px;'>
          <form method='get' action='/jobs' style='display:flex; gap:8px;'>
            <input type='hidden' name='candidate_id' value='{candidate_id or ""}' />
            <input type='text' name='q' placeholder='Search jobs by title or description...' value='{search_query}' style='flex:1;' />
            <button type='submit' style='padding:10px 16px;'>Search</button>
            {"<a href='/jobs" + (f"?candidate_id={candidate_id}" if candidate_id else "") + "' style='padding:10px 12px; color:#6b7280; text-decoration:none;'>Clear</a>" if search_query else ""}
          </form>
        </div>
        <div style='min-width:250px;'>
          <form method='get' action='/jobs' style='display:flex; gap:8px;'>
            <input type='hidden' name='q' value='{search_query}' />
            <select name='candidate_id' style='padding:8px; flex:1;' onchange='this.form.submit()'>
              <option value=''>-- Find relevant jobs for candidate --</option>
              {candidate_options}
            </select>
          </form>
        </div>
        <div>
          <a href='/sync_jobs'><button style='background:#059669;'>Sync from Google Sheet</button></a>
        </div>
      </div>
      <div style='margin-top:8px;' class='muted'>
        {gsheet_status} | Showing {len(scored_jobs) if scored_jobs else len(jobs)} job(s)
        {f" matching '<b>{search_query}</b>'" if search_query else ""}
        {f" | Ranked for <b>{selected_resume['filename']}</b>" if selected_resume else ""}
      </div>
    </div>
    """

    # Render job cards
    jobs_html = ""
    if scored_jobs is not None:
        # Ranked by candidate relevance
        if not scored_jobs:
            jobs_html = "<div class='card muted'>No jobs found.</div>"
        for score, job, match_info in scored_jobs:
            job_data = JobData.from_json(job["extracted_data_json"])
            source = job["source"] if "source" in job.keys() else "manual"
            source_badge = f"<span class='badge' style='background:#dbeafe; color:#1e40af;'>{source}</span>"
            industry_badge = f"<span class='badge industry'>{job_data.industry}</span>" if job_data.industry else ""
            posted = job["posted_date"] if "posted_date" in job.keys() else ""
            posted_html = f"<span class='muted'>Posted: {posted}</span>" if posted else ""
            job_posted_by = job["posted_by"] if "posted_by" in job.keys() else ""
            posted_by_html = f"<span class='muted'>By: <b>{job_posted_by}</b></span>" if job_posted_by else ""

            match_analysis_html = render_match_analysis(match_info)
            job_desc_snippet = snippet(job["description"], 400)

            jobs_html += f"""
            <div class='card'>
              <div class='card-header'>{job["title"]}{industry_badge}{source_badge}</div>
              <div class='score'>Match: {score:.1%} <span class='muted' style='font-weight:normal;'>(Skill: {match_info['match_score']:.0%} | Semantic: {match_info['semantic_score']:.0%})</span></div>
              <div class='meta'>{posted_html}{(' | ' + posted_by_html) if posted_by_html else ''}</div>
              {match_analysis_html}
              <pre class='muted' style='font-family:inherit; margin-top:8px;'>{job_desc_snippet}</pre>
              {render_skills_section("Required Technical", job_data.required_technical_skills, "technical")}
              {render_skills_section("Required Business", job_data.required_business_skills, "business")}
              {render_skills_section("Preferred Skills", job_data.all_preferred_skills(), "preferred")}
            </div>
            """
    else:
        # No candidate selected — just list jobs
        if not jobs:
            jobs_html = "<div class='card muted'>No jobs found. Post jobs via Recruiter Portal or sync from Google Sheets.</div>"
        for job in jobs:
            job_data = JobData.from_json(job["extracted_data_json"])
            source = job["source"] if "source" in job.keys() else "manual"
            source_badge = f"<span class='badge' style='background:#dbeafe; color:#1e40af;'>{source}</span>"
            industry_badge = f"<span class='badge industry'>{job_data.industry}</span>" if job_data.industry else ""
            seniority_badge = f"<span class='badge seniority'>{job_data.seniority_level}</span>" if job_data.seniority_level else ""
            posted = job["posted_date"] if "posted_date" in job.keys() else ""
            posted_html = f"<span class='muted'>Posted: {posted}</span>" if posted else ""
            job_posted_by = job["posted_by"] if "posted_by" in job.keys() else ""
            posted_by_html = f"<span class='muted'>By: <b>{job_posted_by}</b></span>" if job_posted_by else ""

            job_meta_parts = []
            if job_data.department:
                job_meta_parts.append(f"<b>Department:</b> {job_data.department}")
            if job_data.job_type:
                job_meta_parts.append(f"<b>Type:</b> {job_data.job_type}")
            if job_data.experience_years_min or job_data.experience_years_max:
                exp = f"{job_data.experience_years_min or 0}-{job_data.experience_years_max or '?'} years"
                job_meta_parts.append(f"<b>Experience:</b> {exp}")
            job_meta_html = " | ".join(job_meta_parts) if job_meta_parts else ""

            job_desc_snippet = snippet(job["description"], 400)
            jobs_html += f"""
            <div class='card'>
              <div class='card-header'>{job["title"]}{industry_badge}{seniority_badge}{source_badge}</div>
              <div class='meta'>{posted_html}{(' | ' + posted_by_html) if posted_by_html else ''}{(" | " + job_meta_html) if job_meta_html else ""}</div>
              <pre class='muted' style='font-family:inherit; margin-top:8px;'>{job_desc_snippet}</pre>
              {render_skills_section("Required Technical", job_data.required_technical_skills, "technical")}
              {render_skills_section("Required Business", job_data.required_business_skills, "business")}
              {render_skills_section("Preferred Skills", job_data.all_preferred_skills(), "preferred")}
              {f"<div class='summary'><b>Summary:</b> {job_data.summary}</div>" if job_data.summary else ""}
            </div>
            """

    body = f"""
    <h1>Jobs Board</h1>
    <p class='muted'>Browse all jobs, search by keyword, or find relevant jobs for a specific candidate.</p>
    {controls_html}
    {jobs_html}
    """
    return html_page("Jobs Board", body)


@app.get("/debug_llm", response_class=HTMLResponse)
def debug_llm() -> str:
    """Debug endpoint to check LLM configuration."""
    api_key_set = bool(LLM_API_KEY)
    api_key_length = len(LLM_API_KEY) if LLM_API_KEY else 0
    provider = LLM_PROVIDER
    model = LLM_MODEL
    
    llm_available = llm_client and llm_client.is_available()
    provider_name = llm_client.get_provider_name() if llm_client else "Not configured"
    
    # Test Groq API directly if configured
    groq_test_result = ""
    if provider == "groq" and LLM_API_KEY:
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"}
                )
                groq_test_result = f"<p><b>Groq API test:</b> Status {resp.status_code}, Response: {resp.text[:200]}</p>"
        except Exception as e:
            groq_test_result = f"<p><b>Groq API test:</b> Error: {str(e)}</p>"
    
    body = f"""
    <h1>LLM Configuration Debug</h1>
    <div style='font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 5px;'>
        <p><b>LLM_PROVIDER:</b> {provider}</p>
        <p><b>LLM_MODEL:</b> {model}</p>
        <p><b>LLM_API_KEY set:</b> {api_key_set}</p>
        <p><b>LLM_API_KEY length:</b> {api_key_length}</p>
        <p><b>llm_client available:</b> {llm_available}</p>
        <p><b>Provider name:</b> {provider_name}</p>
        {groq_test_result}
    </div>
    <div style='margin-top:20px'>
        <a href='/'><button>Back to Home</button></a>
        <a href='/candidate'><button>Candidate Portal</button></a>
    </div>
    """
    return html_page("LLM Debug", body)


@app.get("/sync_jobs", response_class=HTMLResponse)
def sync_jobs() -> str:
    """Manually trigger Google Sheets sync."""
    if not all([db, extractor, matcher, gsheet_sync]):
        return html_page("Sync Jobs", "", error="System not fully initialized")

    if not gsheet_sync.is_configured():
        body = """
        <h1>Google Sheets Sync</h1>
        <div class='error'>
            <b>Google Sheets is not configured.</b><br><br>
            To set up Google Sheets integration:<br>
            1. Create a Google Cloud project and enable Google Sheets API<br>
            2. Create a service account and download the credentials JSON<br>
            3. Save it as <code>google_credentials.json</code> in the app directory<br>
            4. Share your Google Sheet with the service account email<br>
            5. Set the <code>GOOGLE_SHEET_ID</code> environment variable to your sheet's ID<br><br>
            Your Google Sheet should have columns: <b>Job Title</b>, <b>Job Description</b>, <b>Posted Date</b>
        </div>
        <div style='margin-top:20px'><a href='/jobs'><button>Back to Jobs Board</button></a></div>
        """
        return html_page("Sync Jobs", body)

    try:
        result = gsheet_sync.sync(db, extractor, matcher)
    except Exception as e:
        return html_page("Sync Jobs", f"<a href='/jobs'>Back to Jobs Board</a>", error=f"Sync failed: {str(e)}")

    if result.get("error"):
        body = f"""
        <h1>Google Sheets Sync</h1>
        <div class='warning'>{result['error']}</div>
        <div class='card'>
            <div><b>Expired jobs cleaned up:</b> {result.get('deleted', 0)}</div>
        </div>
        <div style='margin-top:20px'><a href='/jobs'><button>Back to Jobs Board</button></a></div>
        """
        return html_page("Sync Jobs", body)

    # Sheet archive info
    archived = result.get('archived', 0)
    archive_tab = result.get('archive_tab', '')
    sheet_cleanup_error = result.get('sheet_cleanup_error')
    archive_html = ""
    if archived > 0:
        archive_html = f"<div class='success' style='margin-top:12px;'><b>{archived}</b> expired row(s) moved to tab <b>{archive_tab}</b> in Google Sheet.</div>"
    if sheet_cleanup_error:
        archive_html += f"<div class='warning' style='margin-top:8px;'>Sheet cleanup warning: {sheet_cleanup_error}</div>"

    body = f"""
    <h1>Google Sheets Sync Complete</h1>
    <div class='success'>Sync completed successfully!</div>
    <div class='card'>
        <div class='card-header'>Sync Summary</div>
        <div><b>Jobs fetched from sheet:</b> {result.get('total_fetched', 0)}</div>
        <div><b>New jobs added:</b> {result['added']}</div>
        <div><b>Already existing (skipped):</b> {result['skipped']}</div>
        <div><b>Expired jobs cleaned up from DB (>{JOB_RETENTION_DAYS} days):</b> {result['deleted']}</div>
    </div>
    {archive_html}
    <div style='margin-top:20px'><a href='/jobs'><button>Back to Jobs Board</button></a></div>
    """
    return html_page("Sync Jobs", body)


@app.get("/admin", response_class=HTMLResponse)
def admin() -> str:
    if not db:
        return html_page("Admin", "", error="Database not initialized")

    resume_count, job_count = db.get_stats()
    recent_resumes = db.get_recent_resumes(10)
    recent_jobs = db.get_recent_jobs(10)
    llm_status = "online" if llm_client and llm_client.is_available() else "offline"
    provider_name = llm_client.get_provider_name() if llm_client else "Not configured"
    
    # Skills taxonomy stats
    skills_stats = db.get_skills_stats()
    top_skills = db.get_top_skills(15)
    skills_by_category = db.get_skills_by_category()

    resumes_html = ""
    for r in recent_resumes:
        res_data = ResumeData.from_json(r["extracted_data_json"])
        all_skills = res_data.all_skills()[:6]
        resumes_html += f"""
        <div class='card'>
          <div><b>#{r['id']}</b> {r['filename']}</div>
          <div class='skills'>{render_skills_tags(all_skills)}</div>
          <div class='muted'>{r['created_at']}</div>
        </div>
        """
    if not resumes_html:
        resumes_html = "<div class='card muted'>No resumes yet</div>"

    jobs_html = ""
    for j in recent_jobs:
        job_data = JobData.from_json(j["extracted_data_json"])
        all_skills = job_data.all_required_skills()[:6]
        industry_badge = f"<span class='badge industry'>{job_data.industry}</span>" if job_data.industry else ""
        jobs_html += f"""
        <div class='card'>
          <div><b>#{j['id']}</b> {j['title']}{industry_badge}</div>
          <div class='skills'>{render_skills_tags(all_skills, 'required')}</div>
          <div class='muted'>{j['created_at']}</div>
        </div>
        """
    if not jobs_html:
        jobs_html = "<div class='card muted'>No jobs yet</div>"
    
    # Top skills HTML
    top_skills_html = ""
    if top_skills:
        for s in top_skills:
            category_class = s["category"] if s["category"] in ["technical", "business", "domain", "soft", "tools", "certifications"] else ""
            resume_count = s["resume_count"]
            job_count = s["job_count"]
            skill_name = s["skill_name"]
            total_count = resume_count + job_count
            top_skills_html += f"<span class='skill-tag {category_class}' title='Resumes: {resume_count}, Jobs: {job_count}'>{skill_name} ({total_count})</span>"
    else:
        top_skills_html = "<span class='muted'>No skills tracked yet. Upload resumes or post jobs to build the taxonomy.</span>"
    
    # Skills by category HTML
    category_html = ""
    for cat, data in skills_by_category.items():
        category_html += f"<div><b>{cat.title()}:</b> {data['count']} unique skills (R:{data['resume_total'] or 0} / J:{data['job_total'] or 0})</div>"
    if not category_html:
        category_html = "<div class='muted'>No categories yet</div>"

    body = f"""
    <h1>Admin Dashboard</h1>
    <div class='card'>
      <div class='card-header'>System Status</div>
      <div>{provider_name}: <span class='status {llm_status}'>{llm_status}</span></div>
      <div>Database: <code>{DB_PATH}</code></div>
      <div style='margin-top:8px'>
        <b>Resumes:</b> {resume_count} | <b>Jobs:</b> {job_count}
      </div>
    </div>
    
    <div class='card'>
      <div class='card-header'>Skills Taxonomy (Learned from Data)</div>
      <div style='margin-bottom:8px;'>
        <b>Total Unique Skills:</b> {skills_stats['total_unique_skills']} | 
        <b>From Resumes:</b> {skills_stats['skills_from_resumes']} | 
        <b>From Jobs:</b> {skills_stats['skills_from_jobs']}
      </div>
      <h3>Skills by Category</h3>
      {category_html}
      <h3>Top Skills (hover for details)</h3>
      <div class='skills'>{top_skills_html}</div>
    </div>
    
    <h2>Recent Resumes</h2>
    {resumes_html}
    <h2>Recent Jobs</h2>
    {jobs_html}
    """
    return html_page("Admin", body)


# =============================================================================
# Run with: uvicorn start:app --reload --port 8000
# For Render deployment, uses PORT environment variable
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
