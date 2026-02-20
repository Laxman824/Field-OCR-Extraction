# # utils/field_extraction.py

# from typing import Dict, Tuple, Optional
# import re
# from fuzzywuzzy import fuzz

# class FieldExtractor:
#     """Field extraction with enhanced accuracy and validation"""
    
#     def __init__(self, field_patterns: Dict, confidence_threshold: float = 0.6):
#         self.field_patterns = field_patterns
#         self.confidence_threshold = confidence_threshold
#         self.context_cache = {}

#     def extract_with_context(self, text: str, field_name: str, context_window: int = 3) -> Tuple[Optional[str], float]:
#         """Extract field value using contextual information"""
#         if not text:
#             return None, 0.0

#         lines = text.split('\n')
#         pattern = self.field_patterns.get(field_name, {}).get('pattern', '')
        
#         if not pattern:
#             return None, 0.0

#         for i, line in enumerate(lines):
#             if re.search(pattern, line, re.IGNORECASE):
#                 # Get surrounding context
#                 start = max(0, i - context_window)
#                 end = min(len(lines), i + context_window + 1)
#                 context = ' '.join(lines[start:end])
                
#                 # Extract value using context
#                 value, confidence = self._extract_value(context, field_name)
#                 if value and confidence >= self.confidence_threshold:
#                     return value, confidence
        
#         return None, 0.0

#     def _extract_value(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
#         """Extract and validate field value"""
#         field_info = self.field_patterns.get(field_name, {})
#         pattern = field_info.get('pattern', '')
        
#         if not pattern:
#             return None, 0.0

#         # Remove the field label
#         value_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
#         # Apply validation if available
#         validation_pattern = field_info.get('validation')
#         if validation_pattern:
#             match = re.search(validation_pattern, value_text)
#             if match:
#                 return match.group(0), 1.0
        
#         # Apply fuzzy matching
#         if 'example' in field_info:
#             confidence = self._fuzzy_match(value_text, field_info['example'])
#             if confidence >= self.confidence_threshold:
#                 return value_text, confidence
        
#         # Basic extraction
#         words = value_text.split()
#         if words:
#             return ' '.join(words[:5]), 0.6
        
#         return None, 0.0

#     def _fuzzy_match(self, text: str, example: str) -> float:
#         """Calculate fuzzy match confidence score"""
#         return fuzz.ratio(text.lower(), example.lower()) / 100.0


# src/utils/field_extraction.py
"""
Production-grade field extraction engine.

Strategies (tried in order per field):
  1. Direct regex  — scan full text for the extract pattern
  2. Label + value — find label, grab value on same/next line
  3. Key-value     — handle "Label: Value" and "Label\nValue"
  4. Fuzzy label   — fuzzy-match keywords then grab nearby text
  5. Validation    — if a validation regex exists, scan entire text
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Try importing fuzzywuzzy; fall back to basic matching ────────
try:
    from fuzzywuzzy import fuzz

    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False
    logger.info("fuzzywuzzy not installed — fuzzy matching disabled")


class FieldExtractor:
    """Context-aware field extractor with multi-strategy pipeline."""

    def __init__(
        self,
        field_categories: Dict,
        confidence_threshold: float = 0.6,
    ):
        # Flatten categories → {field_name: field_info}
        self.fields: Dict[str, dict] = {}
        for _cat, fields in field_categories.items():
            for name, info in fields.items():
                self.fields[name] = info

        self.threshold = confidence_threshold
        self._compile_patterns()

    # ── Pre-compile regex patterns ──────────────────────────────
    def _compile_patterns(self):
        """Pre-compile all regex for performance."""
        for name, info in self.fields.items():
            # Compile label patterns
            compiled = []
            for p in info.get("patterns", []):
                try:
                    compiled.append(re.compile(p, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Bad regex for {name}: {p} → {e}")
            info["_compiled_patterns"] = compiled

            # Compile extract pattern
            ext = info.get("extract")
            if ext:
                try:
                    info["_compiled_extract"] = re.compile(ext)
                except re.error:
                    info["_compiled_extract"] = None
            else:
                info["_compiled_extract"] = None

            # Compile validation pattern
            val = info.get("validation")
            if val:
                try:
                    info["_compiled_validation"] = re.compile(val)
                except re.error:
                    info["_compiled_validation"] = None
            else:
                info["_compiled_validation"] = None

    # ════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ════════════════════════════════════════════════════════════

    def extract_with_context(
        self,
        text: str,
        field_name: str,
        context_window: int = 3,
    ) -> Tuple[Optional[str], float]:
        """
        Main entry point: extract a single field from text.
        Returns (value, confidence) or (None, 0.0).
        """
        if not text or field_name not in self.fields:
            return None, 0.0

        info = self.fields[field_name]
        lines = text.split("\n")
        clean_text = self._clean_ocr_text(text)
        clean_lines = clean_text.split("\n")

        # ── Strategy 1: Direct extract regex on full text ────
        result = self._try_direct_extract(clean_text, info)
        if result:
            return result

        # ── Strategy 2: Label → value on same / next line ────
        result = self._try_label_value(
            lines, clean_lines, info, context_window
        )
        if result:
            return result

        # ── Strategy 3: Key-value with colon / equals ────────
        result = self._try_key_value(clean_text, info)
        if result:
            return result

        # ── Strategy 4: Fuzzy keyword match ──────────────────
        result = self._try_fuzzy_match(clean_lines, info)
        if result:
            return result

        # ── Strategy 5: Validation scan (last resort) ────────
        result = self._try_validation_scan(clean_text, info)
        if result:
            return result

        return None, 0.0

    def extract_all_fields(
        self,
        text: str,
        selected_fields: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, dict]:
        """Extract all (or selected) fields from text at once."""
        results = {}

        # Sort by priority (higher first)
        sorted_fields = sorted(
            self.fields.items(),
            key=lambda x: x[1].get("priority", 5),
            reverse=True,
        )

        for name, _info in sorted_fields:
            if selected_fields and not selected_fields.get(name, False):
                continue

            value, confidence = self.extract_with_context(text, name)
            if value and confidence >= self.threshold:
                results[name] = {
                    "value": value,
                    "confidence": confidence,
                }

        return results

    # ════════════════════════════════════════════════════════════
    #  EXTRACTION STRATEGIES
    # ════════════════════════════════════════════════════════════

    def _try_direct_extract(
        self, text: str, info: dict
    ) -> Optional[Tuple[str, float]]:
        """Strategy 1: Use the extract regex directly on full text."""
        compiled = info.get("_compiled_extract")
        if not compiled:
            return None

        # For high-specificity patterns (PAN, IFSC, ISIN, Aadhaar, Email)
        # we can confidently grab the first match from anywhere
        val_re = info.get("_compiled_validation")
        if val_re:
            match = compiled.search(text)
            if match:
                value = match.group(0).strip()
                value = self._clean_value(value, info)
                if val_re.match(value):
                    return value, 0.98
        return None

    def _try_label_value(
        self,
        raw_lines: List[str],
        clean_lines: List[str],
        info: dict,
        context_window: int,
    ) -> Optional[Tuple[str, float]]:
        """Strategy 2: Find label line, extract value from same/next lines."""
        patterns = info.get("_compiled_patterns", [])
        hint = info.get("value_hint", "same_line")
        extract_re = info.get("_compiled_extract")

        for line_idx, line in enumerate(clean_lines):
            for pat in patterns:
                match = pat.search(line)
                if not match:
                    continue

                # ── Determine search zone ────────────────────
                if hint == "same_line":
                    # Text after the label on the same line
                    after_label = line[match.end():].strip()
                    after_label = self._strip_separators(after_label)

                    if after_label and len(after_label) > 1:
                        value = self._extract_from_text(
                            after_label, extract_re, info
                        )
                        if value:
                            return value, 0.92

                    # Fallback: check the next line
                    if line_idx + 1 < len(clean_lines):
                        next_line = clean_lines[line_idx + 1].strip()
                        if next_line and not self._is_label_line(next_line):
                            value = self._extract_from_text(
                                next_line, extract_re, info
                            )
                            if value:
                                return value, 0.85

                elif hint == "next_line":
                    if line_idx + 1 < len(clean_lines):
                        next_line = clean_lines[line_idx + 1].strip()
                        value = self._extract_from_text(
                            next_line, extract_re, info
                        )
                        if value:
                            return value, 0.88

                elif hint == "next_lines":
                    # Grab multiple lines (for addresses)
                    collected = []
                    for j in range(
                        line_idx + 1,
                        min(line_idx + 1 + context_window, len(clean_lines)),
                    ):
                        next_l = clean_lines[j].strip()
                        if not next_l:
                            continue
                        if self._is_label_line(next_l):
                            break
                        collected.append(next_l)

                    # Also grab text after label on same line
                    after = line[match.end():].strip()
                    after = self._strip_separators(after)
                    if after and len(after) > 2:
                        collected.insert(0, after)

                    if collected:
                        value = ", ".join(collected)
                        value = self._clean_value(value, info)
                        if value:
                            return value, 0.82

                elif hint == "nearby":
                    # Search within context window
                    start = max(0, line_idx - 1)
                    end = min(len(clean_lines), line_idx + context_window + 1)
                    zone = " ".join(clean_lines[start:end])
                    value = self._extract_from_text(zone, extract_re, info)
                    if value:
                        return value, 0.75

        return None

    def _try_key_value(
        self, text: str, info: dict
    ) -> Optional[Tuple[str, float]]:
        """Strategy 3: Match 'Label : Value' or 'Label = Value' patterns."""
        patterns = info.get("_compiled_patterns", [])
        extract_re = info.get("_compiled_extract")

        for pat in patterns:
            # Look for "Label[: =] Value" on one line
            kv_pattern = re.compile(
                pat.pattern + r"\s*[:=\-|]\s*(.+)",
                re.IGNORECASE,
            )
            match = kv_pattern.search(text)
            if match and match.group(1):
                raw_value = match.group(1).strip()
                # Take only first line of captured text
                raw_value = raw_value.split("\n")[0].strip()
                value = self._extract_from_text(raw_value, extract_re, info)
                if value:
                    return value, 0.90
        return None

    def _try_fuzzy_match(
        self, lines: List[str], info: dict
    ) -> Optional[Tuple[str, float]]:
        """Strategy 4: Fuzzy-match keywords to find the right line."""
        if not HAS_FUZZY:
            return None

        keywords = info.get("keywords", [])
        if not keywords:
            return None

        extract_re = info.get("_compiled_extract")
        best_score = 0
        best_line_idx = -1
        best_keyword = ""

        for idx, line in enumerate(lines):
            line_lower = line.lower().strip()
            if len(line_lower) < 2:
                continue

            for kw in keywords:
                # Check the first portion of the line (likely label area)
                check_portion = line_lower[:60]
                score = fuzz.partial_ratio(kw.lower(), check_portion)
                if score > best_score and score >= 70:
                    best_score = score
                    best_line_idx = idx
                    best_keyword = kw

        if best_line_idx < 0:
            return None

        line = lines[best_line_idx]

        # Try to extract value from the part after the keyword
        kw_pos = line.lower().find(best_keyword.lower())
        if kw_pos >= 0:
            after = line[kw_pos + len(best_keyword):].strip()
            after = self._strip_separators(after)
            if after:
                value = self._extract_from_text(after, extract_re, info)
                if value:
                    conf = min(0.85, best_score / 100.0)
                    return value, conf

        # Try next line
        if best_line_idx + 1 < len(lines):
            next_line = lines[best_line_idx + 1].strip()
            if next_line and not self._is_label_line(next_line):
                value = self._extract_from_text(
                    next_line, extract_re, info
                )
                if value:
                    conf = min(0.78, best_score / 100.0 * 0.9)
                    return value, conf

        return None

    def _try_validation_scan(
        self, text: str, info: dict
    ) -> Optional[Tuple[str, float]]:
        """Strategy 5: Scan entire text with validation regex (last resort)."""
        val_re = info.get("_compiled_validation")
        extract_re = info.get("_compiled_extract")

        if extract_re:
            match = extract_re.search(text)
            if match:
                value = match.group(0).strip()
                value = self._clean_value(value, info)
                # Lower confidence since we found it without label context
                if val_re and val_re.match(value):
                    return value, 0.70
                elif not val_re and len(value) > 2:
                    return value, 0.55

        return None

    # ════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════

    def _extract_from_text(
        self,
        text: str,
        extract_re: Optional[re.Pattern],
        info: dict,
    ) -> Optional[str]:
        """Apply extract regex or return cleaned text."""
        if not text or len(text.strip()) < 1:
            return None

        if extract_re:
            match = extract_re.search(text)
            if match:
                value = match.group(0).strip()
                return self._clean_value(value, info)
            return None

        # No extract regex — return cleaned text (capped by max_length)
        value = text.strip()
        return self._clean_value(value, info) if len(value) > 1 else None

    def _clean_value(self, value: str, info: dict) -> Optional[str]:
        """Clean and truncate an extracted value."""
        if not value:
            return None

        # Remove common noise
        value = re.sub(r"^[\s:=\-|•►]+", "", value)
        value = re.sub(r"[\s:=\-|•►]+$", "", value)
        value = re.sub(r"\s+", " ", value).strip()

        # Remove currency symbols for numeric validation
        # (keep them in the output for Amount fields)
        max_len = info.get("max_length", 200)
        if len(value) > max_len:
            value = value[:max_len].strip()

        # Validate if pattern exists
        val_re = info.get("_compiled_validation")
        if val_re:
            # Strip currency for validation check
            check_val = re.sub(r"[₹Rs\.INR,\s]", "", value)
            if not check_val:
                return None
            # For strict patterns, validate
            if val_re.pattern.startswith("^") and not val_re.match(check_val):
                # Still return if the raw value is reasonable
                if len(value) < 3:
                    return None

        return value if len(value) >= 1 else None

    def _strip_separators(self, text: str) -> str:
        """Remove leading colons, dashes, equals, pipes."""
        return re.sub(r"^[\s:=\-|►•]+", "", text).strip()

    def _is_label_line(self, line: str) -> bool:
        """Check if a line looks like a field label (not a value)."""
        line_stripped = line.strip()
        if not line_stripped:
            return False

        # Check against all compiled patterns
        for _name, info in self.fields.items():
            for pat in info.get("_compiled_patterns", []):
                if pat.search(line_stripped):
                    return True
        return False

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Fix common OCR noise while preserving structure."""
        if not text:
            return ""

        # Fix common OCR misreads
        replacements = {
            "|": "I",       # pipe → I (context dependent)
            "``": '"',
            "''": '"',
            "\u2018": "'",  # smart quotes
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",  # en-dash
            "\u2014": "-",  # em-dash
            "\xa0": " ",    # non-breaking space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Collapse multiple spaces (but keep newlines)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove blank lines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE HELPERS (backward compatibility)
# ═══════════════════════════════════════════════════════════════════

def extract_value(text: str, field: str) -> Tuple[Optional[str], float]:
    """
    Legacy standalone function.
    Creates a temporary extractor and runs extraction.
    """
    from config.field_patterns import FIELD_CATEGORIES

    extractor = FieldExtractor(FIELD_CATEGORIES, confidence_threshold=0.3)
    return extractor.extract_with_context(text, field)


def determine_label(field: str) -> str:
    """Classify a field as question / answer / other."""
    question_fields = {
        "Scheme Name", "Folio Number", "Number of Units", "PAN",
        "Tax Status", "Mobile Number", "Email", "Address", "Date",
        "Bank Account Details", "Date of Journey", "Aadhaar",
        "Bank Account Number", "IFSC Code", "Name", "Amount",
        "NAV", "ISIN", "Date of Birth", "Father's Name",
        "PIN Code", "City", "State", "Gender", "Nationality",
        "Bank Name", "Document Number",
    }
    if field in question_fields:
        return "question"
    elif field == "Signature":
        return "other"
    return "answer"