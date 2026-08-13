"""Default roster and workforce configuration values."""

from datetime import date

MIN_MONTH = date(2025, 4, 1)   # Start app from April 2025

# Reference defaults (used if DB rows missing)
DEFAULT_WORKING_CODES = ["M", "D", "A", "N", "SC", "SSC", "SBY"]
DEFAULT_BANNED_ROSTER_CODES = ["SIC", "SC", "SSC", "AL", "SP", "SPL", "PL", "TOU8", "TOUI"]
DEFAULT_EXCLUDE_FROM_COUNTERS = ["OSS"]
DEFAULT_NON_WORKING_CODES = [
    "OFF", "AL", "PL", "SPL", "TOU8", "TOUI", "OSS", "OFFICE", "WFH", "CTB", "MTG"
]

DEFAULT_ANNOTATION_TYPES = [
    {
        "code": "INFO",
        "label": "Information",
        "category": "Information",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "info,report_exclude",
        "is_active": True,
        "sort_order": 0,
    },
    {
        "code": "EXTS",
        "label": "Short EXT",
        "category": "Extensions",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ext,ext_short",
        "is_active": True,
    },
    {
        "code": "EXTL",
        "label": "Long EXT",
        "category": "Extensions",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ext,ext_long",
        "is_active": True,
    },
    {
        "code": "SWAP",
        "label": "Swap",
        "category": "Swaps",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "swap",
        "is_active": True,
    },
    {
        "code": "A2",
        "label": "A2",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A4",
        "label": "A4",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A6",
        "label": "A6",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "A8",
        "label": "A8",
        "category": "Overtime",
        "allow_suffix": True,
        "suffixes": "MDAN",
        "toil_half_days": 0,
        "tags": "ot,aava",
        "is_active": True,
    },
    {
        "code": "SOAL",
        "label": "SOAL",
        "category": "Overtime",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 0,
        "tags": "ot,soal",
        "is_active": True,
    },
    {
        "code": "TOA8",
        "label": "TOA8 (TOIL +1.0)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 2,
        "tags": "toil",
        "is_active": True,
    },
    {
        "code": "TOAI",
        "label": "TOAI (TOIL +0.5)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 1,
        "tags": "toil",
        "is_active": True,
    },
    {
        "code": "TOAU",
        "label": "TOAU (legacy)",
        "category": "TOIL Accrual",
        "allow_suffix": False,
        "suffixes": "",
        "toil_half_days": 2,
        "tags": "toil",
        "is_active": False,
    },
]

DEFAULT_ROSTER_SETTINGS = {
    "working_codes": DEFAULT_WORKING_CODES,
    "banned_codes": DEFAULT_BANNED_ROSTER_CODES,
    "exclude_from_counters": DEFAULT_EXCLUDE_FROM_COUNTERS,
    "non_working_codes": DEFAULT_NON_WORKING_CODES,
}

OPERATIONAL_CURRENCY_SETTING_KEY = "operational_currency_requirement"
DEFAULT_OPERATIONAL_CURRENCY_REQUIREMENT = {
    "enabled": False,
    "period_type": "rolling_days",
    "period_days": 30,
    "period_months": 1,
    "start_date": "",
    "hours_per_ue": 10,
    "ojti_credit_percent": 25,
}

DEFAULT_ABSENCE_TYPES = [
    {"code": "AL", "label": "Annual leave", "category": "leave", "active": True},
    {"code": "PL", "label": "Parental leave", "category": "leave", "active": True},
    {"code": "SPL", "label": "Special leave", "category": "leave", "active": True},
    {"code": "SC", "label": "Sickness", "category": "sickness", "active": True},
    {"code": "SSC", "label": "Self-certified sickness", "category": "sickness", "active": True},
]
