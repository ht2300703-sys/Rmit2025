"""
End-to-End Defense Pipeline -- Member D (Pham Khang)
He thong phong thu GenAI da tang (Defense-in-Depth Architecture):
- Tang 1: Ingress Guardrail   (Huy & Dung -- model phan loai)
- Tang 2: Prompt Hardening    (Pham Khang -- canary + XML isolation)
- Tang 3: GenAI Inference     (Pham Khang -- engine co fallback)
- Tang 4: Egress Guardrail    (Pham Khang -- PII + canary + policy)
"""
import time
from typing import Dict, Any, Optional, Callable, Tuple

from src.prompt_security import HardenedPromptBuilder
from src.egress_guard import EgressGuardrail
from src.inference import GenAIInferenceEngine


class EndToEndDefensePipeline:
    """
    He thong phong thu da tang ket hop ca 4 module chinh.
    Cho phep truyen ingress_classifier_fn tuy chinh (model hoc may that).
    """

    def __init__(
        self,
        ingress_classifier_fn: Optional[Callable[[str], Tuple[bool, float]]] = None,
        ingress_threshold: float = 0.5,
        llm_engine: Optional[GenAIInferenceEngine] = None,
        verbose: bool = False,
    ):
        """
        Args:
            ingress_classifier_fn: Ham (text) -> (is_malicious: bool, score: float).
                                   None = bo qua tang 1.
            ingress_threshold:     Nguong rui ro de kich hoat chan tai tang 1.
            llm_engine:            GenAI engine; neu None se khoi tao mac dinh.
            verbose:               In log debug.
        """
        self.ingress_classifier = ingress_classifier_fn
        self.ingress_threshold = ingress_threshold
        self.prompt_builder = HardenedPromptBuilder()
        self.llm_engine = llm_engine or GenAIInferenceEngine(verbose=verbose)
        self.egress_guardrail = EgressGuardrail()
        self.verbose = verbose

    def process(
        self,
        user_prompt: str,
        external_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Xu ly mot user prompt qua toan bo pipeline phong thu.

        Returns dict:
            status:           'ALLOWED' | 'BLOCKED' | 'PENDING'
            blocked_at_layer: 'LAYER_1_INGRESS' | 'LAYER_3_EGRESS' | None
            blocked_reason:   Ly do chan (chuoi mo ta).
            ingress_score:    Xac suat rui ro tu classifier.
            canary_token:     Canary da tiem vao prompt.
            response:         Phan hoi cuoi cung (sach).
            latency_ms:       Tong do tre xu ly (ms).
            details:          Chi tiet audit cac tang.
        """
        start_time = time.time()
        result: Dict[str, Any] = {
            "prompt": user_prompt,
            "status": "PENDING",
            "blocked_at_layer": None,
            "blocked_reason": None,
            "ingress_score": 0.0,
            "canary_token": None,
            "response": "",
            "latency_ms": 0.0,
            "details": {},
        }

        # ------------------------------------------------------------------ #
        # TANG 1: INGRESS GUARDRAIL
        # ------------------------------------------------------------------ #
        if self.ingress_classifier is not None:
            is_malicious, risk_score = self.ingress_classifier(user_prompt)
            result["ingress_score"] = float(risk_score)
            if self.verbose:
                print(f"[Pipeline] Tang 1 Ingress: malicious={is_malicious}, score={risk_score:.4f}")

            if is_malicious or risk_score >= self.ingress_threshold:
                result["status"] = "BLOCKED"
                result["blocked_at_layer"] = "LAYER_1_INGRESS"
                result["blocked_reason"] = (
                    f"Ingress classifier triggered (Risk score: {risk_score:.4f} "
                    f">= {self.ingress_threshold:.4f})"
                )
                result["response"] = (
                    "Yeu cau cua ban da bi tu choi boi bo loc an toan dau vao (Ingress Guardrail)."
                )
                result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                return result

        # ------------------------------------------------------------------ #
        # TANG 2: SYSTEM PROMPT HARDENING & CONTEXT ISOLATION
        # ------------------------------------------------------------------ #
        hardened_prompt, canary_token = self.prompt_builder.build_prompt(
            user_prompt=user_prompt,
            external_context=external_context,
        )
        result["canary_token"] = canary_token
        if self.verbose:
            print(f"[Pipeline] Tang 2 Hardening: canary={canary_token}")

        # ------------------------------------------------------------------ #
        # TANG 3: GENAI INFERENCE ENGINE
        # ------------------------------------------------------------------ #
        raw_output = self.llm_engine.generate(hardened_prompt)
        result["details"]["raw_output_length"] = len(raw_output)
        if self.verbose:
            print(f"[Pipeline] Tang 3 Inference: output={raw_output[:80]}...")

        # ------------------------------------------------------------------ #
        # TANG 4: EGRESS GUARDRAIL
        # ------------------------------------------------------------------ #
        egress_audit = self.egress_guardrail.filter(raw_output, canary_token)
        result["details"]["egress_audit"] = egress_audit

        if not egress_audit["passed"]:
            result["status"] = "BLOCKED"
            result["blocked_at_layer"] = "LAYER_3_EGRESS"
            result["blocked_reason"] = egress_audit["blocked_reason"]
            result["response"] = egress_audit["final_text"]
        else:
            result["status"] = "ALLOWED"
            result["response"] = egress_audit["final_text"]

        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        if self.verbose:
            print(f"[Pipeline] Ket qua cuoi: {result['status']} ({result['latency_ms']} ms)")
        return result
