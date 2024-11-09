# src/config/field_patterns.py

FIELD_CATEGORIES = {
    "Personal Information": {
        "PAN": {
            "pattern": r"\b(PAN|Permanent\s+Account\s+Number)\b",
            "description": "Permanent Account Number for tax identification",
            "example": "ABCDE1234F",
            "validation": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
        },
        "Name": {
            "pattern": r"\b(Name|Full\s+Name)\b",
            "description": "Full name of the person",
            "example": "John Doe",
            "validation": r"\b[A-Za-z\s\.]+\b"
        },
        "Tax Status": {
            "pattern": r"\bTax\s+Status\b",
            "description": "Tax category of the investor",
            "example": "Individual, NRI, Company",
            "validation": None
        }
    },
    "Investment Details": {
        "Scheme Name": {
            "pattern": r"\b(Scheme|Plan)\s+Name\b",
            "description": "Name of the investment scheme or plan",
            "example": "Growth Fund, Equity Fund",
            "validation": None
        },
        "Folio Number": {
            "pattern": r"\b(Folio|Account)\s+(Number|No\.?)\b",
            "description": "Unique identifier for your investment account",
            "example": "1234567890",
            "validation": r"\b[A-Za-z0-9]+\b"
        },
        "Number of Units": {
            "pattern": r"\b(Number\s+of\s+Units|Units|Quantity)\b",
            "description": "Number of units held in the investment",
            "example": "100.50",
            "validation": r"\b\d+(\.\d+)?\b"
        }
    },
    "Contact Information": {
        "Mobile Number": {
            "pattern": r"\b(Mobile|Phone|Cell)\s+(Number|No\.?)\b",
            "description": "Contact phone number",
            "example": "+91 9876543210",
            "validation": r"\b(\+\d{1,3}[-.\s]?)?\d{10,14}\b"
        },
        "Email": {
            "pattern": r"\b(Email|E-mail)\b",
            "description": "Email address for correspondence",
            "example": "investor@example.com",
            "validation": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        },
        "Address": {
            "pattern": r"\bAddress\b",
            "description": "Physical address for correspondence",
            "example": "123 Main St, City",
            "validation": None
        }
    },
    "Banking Information": {
        "Bank Account Details": {
            "pattern": r"\b(Bank\s+Account|Account)\s+(Details|Information)\b",
            "description": "Bank account and IFSC information",
            "example": "A/C: 1234567890, IFSC: ABCD0123456",
            "validation": None
        }
    },
    "Date Information": {
        "Date": {
            "pattern": r"\b[Dd]ate\b",
            "description": "Document or transaction date",
            "example": "2024-01-01",
            "validation": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
        },
        "Date of Journey": {
            "pattern": r"\bDate\s+of\s+Journey\b",
            "description": "Travel date information",
            "example": "2024-01-01",
            "validation": r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
        }
    }
}