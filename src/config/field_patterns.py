# # src/config/field_patterns.py

# FIELD_CATEGORIES = {
#     "Personal Information": {
#         "PAN": {
#             "pattern": r"\b(PAN|Permanent\s+Account\s+Number)\b",
#             "description": "Permanent Account Number for tax identification",
#             "example": "ABCDE1234F",
#             "validation": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
#         },
#         "Name": {
#             "pattern": r"\b(Name|Full\s+Name)\b",
#             "description": "Full name of the person",
#             "example": "John Doe",
#             "validation": r"\b[A-Za-z\s\.]+\b"
#         },
#         "Tax Status": {
#             "pattern": r"\bTax\s+Status\b",
#             "description": "Tax category of the investor",
#             "example": "Individual, NRI, Company",
#             "validation": None
#         }
#     },
#     "Investment Details": {
#         "Scheme Name": {
#             "pattern": r"\b(Scheme|Plan)\s+Name\b",
#             "description": "Name of the investment scheme or plan",
#             "example": "Growth Fund, Equity Fund",
#             "validation": None
#         },
#         "Folio Number": {
#             "pattern": r"\b(Folio|Account)\s+(Number|No\.?)\b",
#             "description": "Unique identifier for your investment account",
#             "example": "1234567890",
#             "validation": r"\b[A-Za-z0-9]+\b"
#         },
#         "Number of Units": {
#             "pattern": r"\b(Number\s+of\s+Units|Units|Quantity)\b",
#             "description": "Number of units held in the investment",
#             "example": "100.50",
#             "validation": r"\b\d+(\.\d+)?\b"
#         }
#     },
#     "Contact Information": {
#         "Mobile Number": {
#             "pattern": r"\b(Mobile|Phone|Cell)\s+(Number|No\.?)\b",
#             "description": "Contact phone number",
#             "example": "+91 9876543210",
#             "validation": r"\b(\+\d{1,3}[-.\s]?)?\d{10,14}\b"
#         },
#         "Email": {
#             "pattern": r"\b(Email|E-mail)\b",
#             "description": "Email address for correspondence",
#             "example": "investor@example.com",
#             "validation": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
#         },
#         "Address": {
#             "pattern": r"\bAddress\b",
#             "description": "Physical address for correspondence",
#             "example": "123 Main St, City",
#             "validation": None
#         }
#     },
#     "Banking Information": {
#         "Bank Account Details": {
#             "pattern": r"\b(Bank\s+Account|Account)\s+(Details|Information)\b",
#             "description": "Bank account and IFSC information",
#             "example": "A/C: 1234567890, IFSC: ABCD0123456",
#             "validation": None
#         }
#     },
#     "Date Information": {
#         "Date": {
#             "pattern": r"\b[Dd]ate\b",
#             "description": "Document or transaction date",
#             "example": "2024-01-01",
#             "validation": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
#         },
#         "Date of Journey": {
#             "pattern": r"\bDate\s+of\s+Journey\b",
#             "description": "Travel date information",
#             "example": "2024-01-01",
#             "validation": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
#         }
#     }
# }


# src/config/field_patterns.py
"""
Production-grade field pattern definitions.

Each field has:
  - patterns    : list of regex to find the LABEL in text
  - validation  : regex the extracted VALUE must match
  - extract     : regex to pull the value directly from surrounding text
  - keywords    : fuzzy-match fallback terms
  - value_hint  : "same_line" | "next_line" | "nearby" | "direct"
  - max_length  : truncate extracted value
  - priority    : higher = checked first (breaks ties)
"""

FIELD_CATEGORIES = {
    # ═══════════════════════════════════════════════════════════
    #  PERSONAL INFORMATION
    # ═══════════════════════════════════════════════════════════
    "Personal Information": {
        "PAN": {
            "patterns": [
                r"(?:PAN|Permanent\s*Account\s*(?:Number|No\.?))",
                r"(?:P\.?A\.?N\.?\s*(?:Number|No\.?|Card)?)",
                r"(?:Income\s*Tax\s*PAN)",
            ],
            "extract": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            "validation": r"^[A-Z]{5}[0-9]{4}[A-Z]$",
            "keywords": ["pan", "permanent account", "tax id"],
            "value_hint": "same_line",
            "max_length": 10,
            "priority": 10,
            "description": "Permanent Account Number (tax ID)",
            "example": "ABCDE1234F",
        },
        "Aadhaar": {
            "patterns": [
                r"(?:Aadhaar|Aadhar|UIDAI|UID)\s*(?:Number|No\.?|Card)?",
                r"(?:Unique\s*Identification)",
            ],
            "extract": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
            "validation": r"^\d{4}\s?\d{4}\s?\d{4}$",
            "keywords": ["aadhaar", "aadhar", "uid"],
            "value_hint": "same_line",
            "max_length": 14,
            "priority": 10,
            "description": "Aadhaar identification number",
            "example": "1234 5678 9012",
        },
        "Name": {
            "patterns": [
                r"(?:(?:Full\s*)?Name\s*(?:of\s*(?:the\s*)?(?:Holder|Investor|Applicant|Customer|Client|Person))?)",
                r"(?:Investor\s*Name|Account\s*Holder|Customer\s*Name)",
                r"(?:Mr\.?|Mrs\.?|Ms\.?|Shri|Smt)\s+",
            ],
            "extract": r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4}",
            "validation": r"^[A-Za-z\s\.\-']{2,80}$",
            "keywords": ["name", "holder", "investor", "applicant"],
            "value_hint": "same_line",
            "max_length": 80,
            "priority": 9,
            "description": "Full name of the person",
            "example": "Rajesh Kumar Sharma",
        },
        "Father's Name": {
            "patterns": [
                r"(?:Father'?s?\s*Name|S/O|D/O|W/O|C/O)",
                r"(?:Son\s*of|Daughter\s*of|Wife\s*of|Care\s*of)",
            ],
            "extract": r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,4}",
            "validation": r"^[A-Za-z\s\.\-']{2,80}$",
            "keywords": ["father", "s/o", "d/o", "son of"],
            "value_hint": "same_line",
            "max_length": 80,
            "priority": 7,
            "description": "Father's / Guardian's name",
            "example": "Suresh Kumar Sharma",
        },
        "Date of Birth": {
            "patterns": [
                r"(?:Date\s*of\s*Birth|D\.?O\.?B\.?|DOB|Birth\s*Date)",
            ],
            "extract": r"\b\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}\b",
            "validation": r"^\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}$",
            "keywords": ["birth", "dob", "born"],
            "value_hint": "same_line",
            "max_length": 12,
            "priority": 8,
            "description": "Date of birth",
            "example": "15/08/1990",
        },
        "Gender": {
            "patterns": [
                r"(?:Gender|Sex)",
            ],
            "extract": r"\b(?:Male|Female|Other|M|F|Transgender)\b",
            "validation": r"^(?:Male|Female|Other|M|F|Transgender)$",
            "keywords": ["gender", "sex"],
            "value_hint": "same_line",
            "max_length": 12,
            "priority": 5,
            "description": "Gender",
            "example": "Male",
        },
        "Tax Status": {
            "patterns": [
                r"(?:Tax\s*Status|Category|Investor\s*(?:Type|Category))",
                r"(?:Status\s*(?:of\s*)?(?:Investor|Applicant))",
            ],
            "extract": r"\b(?:Individual|HUF|Company|NRI|Trust|Partnership|Firm|BOI|AOP|Minor)\b",
            "validation": None,
            "keywords": ["tax status", "investor type", "category"],
            "value_hint": "same_line",
            "max_length": 30,
            "priority": 6,
            "description": "Tax category of the investor",
            "example": "Individual",
        },
        "Nationality": {
            "patterns": [
                r"(?:Nationality|Citizenship|Country\s*of\s*(?:Origin|Citizenship))",
            ],
            "extract": r"\b[A-Z][a-zA-Z]{2,20}\b",
            "validation": None,
            "keywords": ["nationality", "citizen"],
            "value_hint": "same_line",
            "max_length": 30,
            "priority": 4,
            "description": "Nationality",
            "example": "Indian",
        },
    },

    # ═══════════════════════════════════════════════════════════
    #  INVESTMENT DETAILS
    # ═══════════════════════════════════════════════════════════
    "Investment Details": {
        "Scheme Name": {
            "patterns": [
                r"(?:Scheme|Plan|Fund)\s*(?:Name)?",
                r"(?:Name\s*of\s*(?:the\s*)?(?:Scheme|Fund|Plan))",
                r"(?:Mutual\s*Fund\s*Scheme)",
            ],
            "extract": None,
            "validation": None,
            "keywords": ["scheme", "fund", "plan", "mutual fund"],
            "value_hint": "same_line",
            "max_length": 120,
            "priority": 8,
            "description": "Investment scheme / fund name",
            "example": "HDFC Equity Growth Fund",
        },
        "Folio Number": {
            "patterns": [
                r"(?:Folio\s*(?:Number|No\.?|#)?)",
                r"(?:Account\s*(?:Number|No\.?|#))",
                r"(?:A/C\s*(?:No\.?|Number|#))",
            ],
            "extract": r"\b[A-Za-z0-9/\-]{4,20}\b",
            "validation": r"^[A-Za-z0-9/\-]{4,20}$",
            "keywords": ["folio", "account number", "a/c"],
            "value_hint": "same_line",
            "max_length": 20,
            "priority": 9,
            "description": "Investment folio/account number",
            "example": "1234567890/12",
        },
        "Number of Units": {
            "patterns": [
                r"(?:(?:Number|No\.?|Qty)\s*(?:of\s*)?Units)",
                r"(?:Units?\s*(?:Held|Balance|Allotted))",
                r"(?:Quantity|Unit\s*Balance)",
            ],
            "extract": r"\b\d{1,12}(?:\.\d{1,4})?\b",
            "validation": r"^\d{1,12}(?:\.\d{1,4})?$",
            "keywords": ["units", "quantity", "balance"],
            "value_hint": "same_line",
            "max_length": 20,
            "priority": 8,
            "description": "Number of units held",
            "example": "1234.5678",
        },
        "NAV": {
            "patterns": [
                r"(?:NAV|Net\s*Asset\s*Value)",
                r"(?:NAV\s*(?:Per\s*Unit|Value|Price))",
            ],
            "extract": r"(?:₹|Rs\.?|INR)?\s*\d{1,8}(?:\.\d{1,4})?",
            "validation": r"^\d{1,8}(?:\.\d{1,4})?$",
            "keywords": ["nav", "net asset value"],
            "value_hint": "same_line",
            "max_length": 20,
            "priority": 7,
            "description": "Net Asset Value per unit",
            "example": "₹45.2350",
        },
        "Amount": {
            "patterns": [
                r"(?:(?:Total\s*)?(?:Amount|Value|Worth|Cost))",
                r"(?:Market\s*Value|Current\s*Value|Investment\s*Value)",
                r"(?:₹|Rs\.?|INR)\s*\d",
            ],
            "extract": r"(?:₹|Rs\.?|INR)?\s*[\d,]{1,15}(?:\.\d{1,2})?",
            "validation": r"^[\d,]{1,15}(?:\.\d{1,2})?$",
            "keywords": ["amount", "value", "worth", "total"],
            "value_hint": "same_line",
            "max_length": 25,
            "priority": 7,
            "description": "Monetary amount / value",
            "example": "₹1,23,456.78",
        },
        "ISIN": {
            "patterns": [
                r"(?:ISIN\s*(?:Number|No\.?|Code)?)",
            ],
            "extract": r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b",
            "validation": r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$",
            "keywords": ["isin"],
            "value_hint": "same_line",
            "max_length": 12,
            "priority": 8,
            "description": "ISIN code for the security",
            "example": "INE001A01036",
        },
    },

    # ═══════════════════════════════════════════════════════════
    #  CONTACT INFORMATION
    # ═══════════════════════════════════════════════════════════
    "Contact Information": {
        "Mobile Number": {
            "patterns": [
                r"(?:Mobile|Phone|Cell|Contact|Tel)\s*(?:Number|No\.?|#)?",
                r"(?:Mob\.?|Ph\.?)\s*(?:No\.?)?",
            ],
            "extract": r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}",
            "validation": r"^(?:\+91[\s\-]?)?[6-9]\d{9}$",
            "keywords": ["mobile", "phone", "contact", "cell"],
            "value_hint": "same_line",
            "max_length": 15,
            "priority": 8,
            "description": "Mobile / phone number",
            "example": "+91 98765 43210",
        },
        "Email": {
            "patterns": [
                r"(?:E[\-\s]?mail\s*(?:ID|Address)?)",
                r"(?:Email\s*(?:Id|Address)?)",
            ],
            "extract": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            "validation": r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$",
            "keywords": ["email", "e-mail", "mail id"],
            "value_hint": "same_line",
            "max_length": 80,
            "priority": 9,
            "description": "Email address",
            "example": "investor@example.com",
        },
        "Address": {
            "patterns": [
                r"(?:(?:Correspondence|Permanent|Mailing|Postal|Residential)\s*)?Address",
                r"(?:Addr\.?|Res\.?\s*Address)",
            ],
            "extract": None,
            "validation": None,
            "keywords": ["address", "addr", "residence"],
            "value_hint": "next_lines",
            "max_length": 250,
            "priority": 6,
            "description": "Postal / residential address",
            "example": "123, MG Road, Bangalore 560001",
        },
        "PIN Code": {
            "patterns": [
                r"(?:PIN\s*(?:Code)?|Postal\s*Code|Zip\s*Code)",
            ],
            "extract": r"\b[1-9]\d{5}\b",
            "validation": r"^[1-9]\d{5}$",
            "keywords": ["pin", "pincode", "postal code", "zip"],
            "value_hint": "same_line",
            "max_length": 6,
            "priority": 7,
            "description": "6-digit PIN code",
            "example": "560001",
        },
        "City": {
            "patterns": [
                r"(?:City|Town|Place)",
            ],
            "extract": r"\b[A-Z][a-zA-Z\s]{2,30}\b",
            "validation": None,
            "keywords": ["city", "town"],
            "value_hint": "same_line",
            "max_length": 40,
            "priority": 5,
            "description": "City / Town",
            "example": "Bangalore",
        },
        "State": {
            "patterns": [
                r"(?:State|Province)",
            ],
            "extract": r"\b[A-Z][a-zA-Z\s]{2,30}\b",
            "validation": None,
            "keywords": ["state", "province"],
            "value_hint": "same_line",
            "max_length": 40,
            "priority": 5,
            "description": "State / Province",
            "example": "Karnataka",
        },
    },

    # ═══════════════════════════════════════════════════════════
    #  BANKING INFORMATION
    # ═══════════════════════════════════════════════════════════
    "Banking Information": {
        "Bank Name": {
            "patterns": [
                r"(?:Bank\s*(?:Name)?)",
                r"(?:Name\s*of\s*(?:the\s*)?Bank)",
            ],
            "extract": r"[A-Z][A-Za-z\s&\.]{3,50}(?:Bank|Ltd|Limited)?",
            "validation": None,
            "keywords": ["bank name", "banker"],
            "value_hint": "same_line",
            "max_length": 60,
            "priority": 7,
            "description": "Name of the bank",
            "example": "State Bank of India",
        },
        "Bank Account Number": {
            "patterns": [
                r"(?:(?:Bank\s*)?(?:Account|A/C)\s*(?:Number|No\.?|#))",
                r"(?:Savings\s*A/C|Current\s*A/C)",
            ],
            "extract": r"\b\d{9,18}\b",
            "validation": r"^\d{9,18}$",
            "keywords": ["account number", "a/c no", "bank account"],
            "value_hint": "same_line",
            "max_length": 18,
            "priority": 9,
            "description": "Bank account number",
            "example": "1234567890123",
        },
        "IFSC Code": {
            "patterns": [
                r"(?:IFSC\s*(?:Code)?|IFS\s*Code)",
                r"(?:MICR|Branch\s*Code)",
            ],
            "extract": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "validation": r"^[A-Z]{4}0[A-Z0-9]{6}$",
            "keywords": ["ifsc", "ifs code", "branch code"],
            "value_hint": "same_line",
            "max_length": 11,
            "priority": 9,
            "description": "IFSC code of the bank branch",
            "example": "SBIN0001234",
        },
        "Bank Account Details": {
            "patterns": [
                r"(?:Bank\s*(?:Account\s*)?Details|Banking\s*Details|Bank\s*Information)",
            ],
            "extract": None,
            "validation": None,
            "keywords": ["bank details", "banking details", "bank information"],
            "value_hint": "next_lines",
            "max_length": 100,
            "priority": 6,
            "description": "Combined bank account + IFSC",
            "example": "A/C: 1234567890, IFSC: SBIN0001234",
        },
    },

    # ═══════════════════════════════════════════════════════════
    #  DATE INFORMATION
    # ═══════════════════════════════════════════════════════════
    "Date Information": {
        "Date": {
            "patterns": [
                r"(?<!\w)[Dd]ate\b(?!\s*of\s*[Bb]irth)",
                r"(?:As\s*on|As\s*of|Dated|Dt\.?)",
            ],
            "extract": r"\b\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}\b",
            "validation": None,
            "keywords": ["date", "dated", "as on"],
            "value_hint": "same_line",
            "max_length": 12,
            "priority": 6,
            "description": "Document / transaction date",
            "example": "15/01/2024",
        },
        "Date of Journey": {
            "patterns": [
                r"(?:Date\s*of\s*Journey|Travel\s*Date|Journey\s*Date)",
                r"(?:Departure\s*Date|Boarding\s*Date)",
            ],
            "extract": r"\b\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4}\b",
            "validation": None,
            "keywords": ["journey", "travel date", "departure"],
            "value_hint": "same_line",
            "max_length": 12,
            "priority": 7,
            "description": "Travel / journey date",
            "example": "20/03/2024",
        },
    },

    # ═══════════════════════════════════════════════════════════
    #  DOCUMENT INFORMATION
    # ═══════════════════════════════════════════════════════════
    "Document Information": {
        "Document Number": {
            "patterns": [
                r"(?:Document|Doc|Reference|Ref)\s*(?:Number|No\.?|#)",
                r"(?:Certificate\s*No\.?|Serial\s*No\.?)",
            ],
            "extract": r"\b[A-Z0-9/\-]{4,25}\b",
            "validation": None,
            "keywords": ["document number", "ref no", "certificate"],
            "value_hint": "same_line",
            "max_length": 25,
            "priority": 6,
            "description": "Document / reference number",
            "example": "REF/2024/001234",
        },
        "Signature": {
            "patterns": [
                r"(?:Signature|Sign|Authorized\s*Signatory)",
            ],
            "extract": None,
            "validation": None,
            "keywords": ["signature", "sign", "authorized"],
            "value_hint": "nearby",
            "max_length": 50,
            "priority": 2,
            "description": "Signature field marker",
            "example": "Authorized Signatory",
        },
    },
}


# ── Utility: flatten categories → flat dict ─────────────────────
def get_flat_fields() -> dict:
    """Return {field_name: field_info} without category nesting."""
    flat = {}
    for _cat, fields in FIELD_CATEGORIES.items():
        for name, info in fields.items():
            flat[name] = info
    return flat


# ── Utility: get all field names ─────────────────────────────────
def get_all_field_names() -> list:
    """Return a sorted list of every field name."""
    return sorted(get_flat_fields().keys())