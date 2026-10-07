from dataclasses import dataclass, field
from enum import Enum
from typing import List
import numpy as np

class Decision(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"

@dataclass
class RiskEvaluation:
    transaction_id: str
    decision: Decision
    # overall composite risk score-> (0 - 100)
    risk_score: float
    # model outputs
    fraud_probability: float
    anomaly_score: float
    # scenario model output
    predicted_scenario: str
    # explainability
    triggers: List[str] = field(default_factory=list)
    # recommended downstream action
    action_notes: str = ""

class UPIRiskEngine:
    """
    Decision engine for real-time UPI transaction risk evaluation.

    The engine combines:
    1. Supervised fraud probability
    2. Unsupervised anomaly risk
    3. Contextual risk signals
    4. Fraud scenario prediction

    Important:
        risk_score is a composite risk indicator.
        The final ALLOW / REVIEW / BLOCK decision is still
        determined by explicit business rules.
        """

    def __init__(self, block_threshold: float = 0.75, 
                 review_threshold: float = 0.35,
                 anomaly_threshold: float = 0.65,
                 severe_anomaly_threshold: float = 0.80,
                 high_amount_threshold: float = 15000.0,
                 # Risk score weights
                 fraud_weight: float = 0.60,
                 anomaly_weight: float = 0.30,
                 context_weight: float = 0.10):
       
        # Validate configuration
        if not 0 <= review_threshold <= 1:
            raise ValueError("review_threshold must be between 0 and 1.")

        if not 0 <= block_threshold <= 1:
            raise ValueError("block_threshold must be between 0 and 1.")

        if review_threshold >= block_threshold:
            raise ValueError("review_threshold must be lower than block_threshold.")

        if not 0 <= anomaly_threshold <= 1:
            raise ValueError("anomaly_threshold must be between 0 and 1.")

        if not 0 <= severe_anomaly_threshold <= 1:
            raise ValueError("severe_anomaly_threshold must be between 0 and 1.")

        if high_amount_threshold < 0:
            raise ValueError("high_amount_threshold cannot be negative.")

        # weights should sum to 1
        total_weight = (fraud_weight + anomaly_weight + context_weight)

        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Risk weights must sum to 1.0. "
                             f"Current sum = {total_weight:.3f}")

        # Store configuration
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold

        self.anomaly_threshold = anomaly_threshold
        self.severe_anomaly_threshold = severe_anomaly_threshold

        self.high_amount_threshold = high_amount_threshold

        self.fraud_weight = fraud_weight
        self.anomaly_weight = anomaly_weight
        self.context_weight = context_weight

    # Validation
    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        """Validate a probability-like model output."""

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. "
                             f"Received {value}.")

        return value

    @staticmethod
    def _validate_anomaly_score(raw_score: float) -> float:
        """Validate raw Isolation Forest decision_function output."""

        raw_score = float(raw_score)

        if not np.isfinite(raw_score):
            raise ValueError("raw_anomaly_score must be a finite number.")

        return raw_score
    
    # Anomaly score normalization
    def _normalize_anomaly_score(self,raw_score: float) -> float:
        """
        Convert Isolation Forest decision_function score into
        a normalized anomaly-risk score between 0 and 1.

        Isolation Forest:
            positive score -> more normal
            negative score -> more anomalous

        Therefore, the score is inverted.
        IMPORTANT: This is a risk score, NOT a probability.
        """

        raw_score = self._validate_anomaly_score(raw_score)

        # negative raw scores should produce higher risk
        anomaly_risk = 1.0 / (1.0 + np.exp(np.clip(raw_score * 5.0, -50, 50)))

        return float(np.clip(anomaly_risk, 0.0, 1.0))

  
    # Context score

    @staticmethod
    def _calculate_context_score(device_is_new: bool,location_mismatch: bool) -> float:
        """
        Convert contextual signals into a normalized 0-1 risk score.

        Current design:
            normal                         = 0.0
            new device                     = 0.5
            location mismatch              = 0.5
            both signals                   = 1.0
        """

        context_score = 0.0

        if device_is_new:
            context_score += 0.5

        if location_mismatch:
            context_score += 0.5

        return float(np.clip(context_score, 0.0, 1.0))

    # Composite risk score
    def _calculate_composite_risk(self, fraud_probability: float, anomaly_risk: float, context_score: float) -> float:
        """
        Calculate overall risk score from 0-100.

        Current weighting:
            Fraud probability  = 60%
            Anomaly risk       = 30%
            Context risk       = 10%
        """

        composite = (fraud_probability * self.fraud_weight 
                     + anomaly_risk * self.anomaly_weight
                     + context_score * self.context_weight)

        return float(np.clip(composite * 100.0, 0.0, 100.0))

    # Main evaluation
    def evaluate(self, transaction_id: str, amount: float, fraud_prob: float, raw_anomaly_score: float,
                 predicted_scenario: str, device_is_new: bool = False, location_mismatch: bool = False) -> RiskEvaluation:
        """
        Evaluate a transaction and return:
            ALLOW
            REVIEW
            BLOCK

        The method is intentionally deterministic so that the same
        model outputs and contextual signals produce the same decision.
        """

        # Validate inputs

        if not transaction_id:
            raise ValueError("transaction_id cannot be empty.")

        amount = float(amount)

        if not np.isfinite(amount):
            raise ValueError("amount must be a finite number.")

        if amount < 0:
            raise ValueError("amount cannot be negative.")

        fraud_prob = self._validate_probability(fraud_prob,"fraud_prob")
        raw_anomaly_score = self._validate_anomaly_score(raw_anomaly_score)
        predicted_scenario = str(predicted_scenario or "Clean")

        # Convert model outputs
        anomaly_risk = self._normalize_anomaly_score(raw_anomaly_score)

        context_score = self._calculate_context_score(device_is_new=device_is_new,location_mismatch=location_mismatch)

        composite_risk = self._calculate_composite_risk(fraud_probability=fraud_prob,
                                                        anomaly_risk=anomaly_risk,
                                                        context_score=context_score)
        
        # explainability triggers
        triggers: List[str] = []

        # BLOCK RULES
        # Rule 1 -> Very high supervised fraud probability
        if fraud_prob >= self.block_threshold:
            triggers.append(f"High fraud probability "
                            f"({fraud_prob:.2f}) >= "
                            f"block threshold "
                            f"({self.block_threshold:.2f})")

        # Rule 2 -> Strong anomaly + significant fraud probability
        # This prevents a very unusual but legitimate transaction
        # from being automatically blocked based only on anomaly.
        if (anomaly_risk >= self.severe_anomaly_threshold and fraud_prob >= 0.50):
            triggers.append(f"Severe behavioral anomaly "
                            f"({anomaly_risk:.2f}) combined with "
                            f"high fraud probability ({fraud_prob:.2f})")

        # BLOCK decision
        if any([fraud_prob >= self.block_threshold,
                (anomaly_risk >= self.severe_anomaly_threshold and fraud_prob >= 0.50)]):

            return RiskEvaluation(
                transaction_id=transaction_id,
                decision=Decision.BLOCK,
                risk_score=composite_risk,
                fraud_probability=fraud_prob,
                anomaly_score=anomaly_risk,
                predicted_scenario=predicted_scenario,
                triggers=triggers,
                action_notes=("Auto-decline recommended. "
                              "Publish to the blocked-transaction topic "
                              "and route the event to the fraud investigation workflow."))

        # REVIEW RULES
        review_triggers: List[str] = []

        # Rule 3 -> Moderate fraud probability
        if fraud_prob >= self.review_threshold:

            review_triggers.append(f"Moderate fraud probability "
                                   f"({fraud_prob:.2f}) >= "
                                   f"review threshold "
                                   f"({self.review_threshold:.2f})")

        # Rule 4 -> Strong anomaly even if supervised model is uncertain
        if anomaly_risk >= self.anomaly_threshold:
            review_triggers.append(f"High behavioral anomaly "
                                   f"({anomaly_risk:.2f}) >= "
                                   f"anomaly threshold "
                                f"({self.anomaly_threshold:.2f})")

        # Rule 5 -> High-value transaction + contextual signal
        if (amount >= self.high_amount_threshold and (device_is_new or location_mismatch)):
            context_reasons = []

            if device_is_new:
                context_reasons.append("new device")

            if location_mismatch:
                context_reasons.append("location mismatch")

            review_triggers.append(f"High-value transaction "
                                   f"(₹{amount:,.2f}) with "
                                   f"{' and '.join(context_reasons)}")

        # Rule 6 -> Suspicious scenario prediction
        clean_scenarios = {"Clean", "None", "No Fraud Detected", "No_Fraud_Detected", "No Fraud"}

        if predicted_scenario not in clean_scenarios:
            review_triggers.append(f"Fraud scenario detected: "
                                   f"'{predicted_scenario}'")

      
        # REVIEW decision
        if review_triggers:
            triggers.extend(review_triggers)

            return RiskEvaluation(
                transaction_id=transaction_id,
                decision=Decision.REVIEW,
                risk_score=composite_risk,
                fraud_probability=fraud_prob,
                anomaly_score=anomaly_risk,
                predicted_scenario=predicted_scenario,
                triggers=triggers,
                action_notes=("Step-up authentication recommended. "
                              "If verification fails, route the transaction "
                              "to the fraud review workflow."))

        # ALLOW
        return RiskEvaluation(
            transaction_id=transaction_id,
            decision=Decision.ALLOW,
            risk_score=composite_risk,
            fraud_probability=fraud_prob,
            anomaly_score=anomaly_risk,
            predicted_scenario="Clean",
            triggers=["Passed all configured risk checks"],
            action_notes=("Transaction approved for normal processing."))