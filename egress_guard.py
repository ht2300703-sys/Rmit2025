"""
Egress Guardrail -- Member D (Pham Khang)
Loc PII, phat hien ro ri Canary Token, kiem tra chinh sach an toan dau ra.
"""
import re
from typing import Dict, Any, Tuple, Optional


class PIIScrubber:
    """
    Bo loc va an danh thong tin ca nhan (PII) trong phan hoi LLM.
    Ho tro: SDT Viet Nam, CCCD/CMND, Email, API Keys.
    """
    PATTERNS = {
        "VN_PHONE": re.compile(r"\b(0|\+84)(3[2-9]|5[6-9]|7[06-9]|8[0-9]|9[0-9])\d{7}\b"),
        "VN_CCCD_12": re.compile(r"\b\d{12}\b"),
        "VN_CMND_9": re.compile(r"\b\d{9}\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "API_KEY": re.compile(
            r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}"
            r"|Bearer\s+[A-Za-z0-9\-_]{20,}|eyJ[A-Za-z0-9\-_]{20,})\b"
        ),
    }

    @classmethod
    def scrub(cls, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Thay the cac PII bang placeholder [PII:TYPE_REDACTED].
        Returns: (scrubbed_text, stats_dict).
        """
        scrubbed = text
        stats: Dict[str, int] = {}
        for pii_type, pattern in cls.PATTERNS.items():
            matches = pattern.findall(scrubbed)
            if matches:
                stats[pii_type] = len(matches)
                scrubbed = pattern.sub(f"[PII:{pii_type}_REDACTED]", scrubbed)
        return scrubbed, stats


class CanaryLeakDetector:
    """
    Kiem tra ro ri ma Canary Token hoac trich xuat System Prompt.
    Neu ma bi mat xuat hien o dau ra, LLM da bi jailbreak.
    """
    KNOWN_CANARY_PATTERNS = [
        re.compile(r"CANARY_SEC_[A-F0-9]{8}", re.IGNORECASE),
        re.compile(r"DEMO_CANARY_VIETSHIELD_\d{4}", re.IGNORECASE),
    ]

    @classmethod
    def check_leak(cls, output_text: str, canary_token: Optional[str] = None) -> bool:
        """
        Tra ve True neu phat hien ro ri canary token.
        Kiem tra ca token cu the, cac pattern canary chung,
        va ca suffix bi mat (chong jailbreak yeu cau dich, danh van, chen khoang trang).
        """
        if not output_text:
            return False
            
        clean_output = re.sub(r"[\s\-_.:]+", "", output_text).upper()
        
        # Check specific token & its suffix
        if canary_token:
            clean_canary = re.sub(r"[\s\-_.:]+", "", canary_token).upper()
            if clean_canary in clean_output:
                return True
            # Check 8-char secret suffix (e.g. A1B2C3D4 from CANARY_SEC_A1B2C3D4)
            if "_" in canary_token:
                suffix = canary_token.split("_")[-1].upper()
                if len(suffix) >= 6 and suffix in clean_output:
                    return True
                    
        # Check known patterns
        for pattern in cls.KNOWN_CANARY_PATTERNS:
            if pattern.search(output_text):
                return True
        return False


class SafetyPolicyCompliance:
    """
    Kiem tra noi dung dau ra co chua huong dan tan cong mang,
    ma doc hoac noi dung nguy hiem hay khong.
    """
    DANGEROUS_PATTERNS = [
        re.compile(r"(che tao bom|vu khi hoa hoc|tong hop chat doc)", re.IGNORECASE),
        re.compile(r"(viet ma doc|reverse shell|tan cong ddos|bypass antivirus|trojan horse)", re.IGNORECASE),
        re.compile(r"(sql injection payload|xss exploit script|cve-\d{4}-\d{4,})", re.IGNORECASE),
        re.compile(r"(\[dangerous_exploit_payload\]|dan mode activated)", re.IGNORECASE),
    ]

    @classmethod
    def is_violating(cls, text: str) -> Tuple[bool, str]:
        """
        Tra ve (True, matched_string) neu phat hien vi pham chinh sach.
        Tra ve (False, '') neu an toan.
        """
        for pattern in cls.DANGEROUS_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, match.group(0)
        return False, ""


class EgressGuardrail:
    """
    Lop bao ve Egress hoan chinh:
    1. Quet Canary Token Leak (chong trich xuat bi mat he thong).
    2. Quet & Che giau PII (CCCD, SDT, Email, API Keys).
    3. Quet vi pham chinh sach an toan & ma doc.
    4. Tra ve Safe Fallback neu phat hien vi pham bao mat.
    """
    SAFE_FALLBACK_MESSAGE = (
        "Yeu cau cua ban khong the hoan tat do he thong phat hien phan hoi "
        "vi pham chinh sach an toan thong tin hoac ro ri du lieu bao mat "
        "(Egress Policy Triggered)."
    )

    def __init__(self):
        self.pii_scrubber = PIIScrubber()
        self.canary_detector = CanaryLeakDetector()
        self.policy_compliance = SafetyPolicyCompliance()

    def filter(self, raw_output: str, canary_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Loc phan hoi LLM qua 3 tang bao ve.
        Returns: audit_log dict voi 'passed', 'blocked_reason', 'final_text', v.v.
        """
        audit_log: Dict[str, Any] = {
            "passed": False,
            "blocked_reason": None,
            "canary_leaked": False,
            "pii_redacted": {},
            "policy_violation": None,
            "final_text": self.SAFE_FALLBACK_MESSAGE,
        }

        # 1. Kiem tra ro ri Canary Token
        if self.canary_detector.check_leak(raw_output, canary_token):
            audit_log["canary_leaked"] = True
            audit_log["blocked_reason"] = "CANARY_TOKEN_LEAK_DETECTED"
            return audit_log

        # 2. Kiem tra noi dung nguy hiem
        is_violated, pattern_str = self.policy_compliance.is_violating(raw_output)
        if is_violated:
            audit_log["policy_violation"] = pattern_str
            audit_log["blocked_reason"] = f"SAFETY_POLICY_VIOLATION: {pattern_str}"
            return audit_log

        # 3. Loc PII
        clean_text, pii_stats = self.pii_scrubber.scrub(raw_output)
        audit_log["pii_redacted"] = pii_stats
        audit_log["passed"] = True
        audit_log["final_text"] = clean_text
        return audit_log
