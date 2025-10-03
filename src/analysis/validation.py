"""
Validation and robustness testing system for ethical agent simulations.

This module provides comprehensive validation mechanisms to ensure
agent and society behaviors are realistic and consistent.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class ValidationType(Enum):
    """Types of validation checks available."""

    AGENT_CONSISTENCY = "agent_consistency"
    SOCIETY_STABILITY = "society_stability"
    DECISION_PLAUSIBILITY = "decision_plausibility"
    LEARNING_VALIDITY = "learning_validity"
    NETWORK_INTEGRITY = "network_integrity"
    SCENARIO_RESPONSE = "scenario_response"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: datetime


class AgentValidator:
    """Validates individual agent behavior and consistency."""

    @staticmethod
    def validate_belief_consistency(agent) -> ValidationResult:
        """Check if agent's beliefs are internally consistent."""
        if not hasattr(agent, "beliefs") or not agent.beliefs.beliefs:
            return ValidationResult(
                check_name="belief_consistency",
                validation_type=ValidationType.AGENT_CONSISTENCY,
                severity=ValidationSeverity.WARNING,
                passed=False,
                score=0.0,
                message="Agent has no beliefs to validate",
                details={"belief_count": 0},
                timestamp=datetime.now(),
            )

        beliefs = agent.beliefs.beliefs
        belief_values = list(beliefs.values())

        # Check for extreme contradictions
        contradictions = 0
        total_pairs = 0

        for i, (belief1, value1) in enumerate(beliefs.items()):
            for belief2, value2 in list(beliefs.items())[i + 1 :]:
                total_pairs += 1
                # Check for semantic contradictions (this is simplified)
                if AgentValidator._are_contradictory(belief1, belief2, value1, value2):
                    contradictions += 1

        if total_pairs == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (contradictions / total_pairs)

        passed = consistency_score >= 0.7  # Threshold for consistency
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="belief_consistency",
            validation_type=ValidationType.AGENT_CONSISTENCY,
            severity=severity,
            passed=passed,
            score=consistency_score,
            message=f"Belief consistency score: {consistency_score:.2f}",
            details={
                "belief_count": len(beliefs),
                "contradictions": contradictions,
                "total_pairs": total_pairs,
                "belief_range": (
                    (min(belief_values), max(belief_values)) if belief_values else (0, 0)
                ),
            },
            timestamp=datetime.now(),
        )

    @staticmethod
    def validate_decision_plausibility(agent) -> ValidationResult:
        """Check if agent's decisions are plausible given its beliefs."""
        if not hasattr(agent, "decision_history") or not agent.decision_history:
            return ValidationResult(
                check_name="decision_plausibility",
                validation_type=ValidationType.DECISION_PLAUSIBILITY,
                severity=ValidationSeverity.INFO,
                passed=True,
                score=1.0,
                message="No decisions to validate",
                details={"decision_count": 0},
                timestamp=datetime.now(),
            )

        recent_decisions = agent.decision_history[-10:]
        plausibility_scores = []

        for decision in recent_decisions:
            # Check if decision aligns with beliefs
            decision_value = decision.get("value", 0.5)
            scenario = decision.get("scenario", "")

            # Simple plausibility check based on decision consistency
            if hasattr(agent, "beliefs") and agent.beliefs.beliefs:
                relevant_beliefs = [
                    v
                    for k, v in agent.beliefs.beliefs.items()
                    if any(keyword in scenario.lower() for keyword in k.lower().split("_"))
                ]

                if relevant_beliefs:
                    avg_belief = np.mean(relevant_beliefs)
                    # Plausibility based on alignment with relevant beliefs
                    plausibility = 1.0 - abs(decision_value - (avg_belief + 1) / 2)
                    plausibility_scores.append(plausibility)
                else:
                    plausibility_scores.append(0.8)  # Neutral if no relevant beliefs
            else:
                plausibility_scores.append(0.8)  # Neutral if no beliefs

        avg_plausibility = np.mean(plausibility_scores) if plausibility_scores else 0.0
        passed = avg_plausibility >= 0.6
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="decision_plausibility",
            validation_type=ValidationType.DECISION_PLAUSIBILITY,
            severity=severity,
            passed=passed,
            score=avg_plausibility,
            message=f"Decision plausibility score: {avg_plausibility:.2f}",
            details={
                "decision_count": len(recent_decisions),
                "plausibility_scores": plausibility_scores,
            },
            timestamp=datetime.now(),
        )

    @staticmethod
    def validate_learning_progress(agent) -> ValidationResult:
        """Check if agent's learning appears valid and not erratic."""
        if not hasattr(agent, "decision_history") or len(agent.decision_history) < 5:
            return ValidationResult(
                check_name="learning_progress",
                validation_type=ValidationType.LEARNING_VALIDITY,
                severity=ValidationSeverity.INFO,
                passed=True,
                score=1.0,
                message="Insufficient data for learning validation",
                details={
                    "decision_count": (
                        len(agent.decision_history) if hasattr(agent, "decision_history") else 0
                    )
                },
                timestamp=datetime.now(),
            )

        decisions = agent.decision_history
        decision_values = [d.get("value", 0.5) for d in decisions]

        # Check for learning patterns
        learning_score = 0.0

        # 1. Check for reasonable variance (not too chaotic, not too static)
        variance = np.var(decision_values)
        if 0.01 <= variance <= 0.25:  # Reasonable range
            learning_score += 0.3

        # 2. Check for trend stability (not oscillating wildly)
        if len(decision_values) >= 10:
            recent_values = decision_values[-10:]
            older_values = (
                decision_values[-20:-10] if len(decision_values) >= 20 else decision_values[:-10]
            )

            if older_values:
                recent_var = np.var(recent_values)
                older_var = np.var(older_values)
                # Learning should reduce variance over time
                if recent_var <= older_var * 1.2:  # Allow some increase
                    learning_score += 0.4

        # 3. Check for confidence patterns
        confidences = [d.get("confidence", 0.5) for d in decisions[-10:]]
        if confidences and np.mean(confidences) >= 0.4:  # Reasonable confidence
            learning_score += 0.3

        passed = learning_score >= 0.5
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="learning_progress",
            validation_type=ValidationType.LEARNING_VALIDITY,
            severity=severity,
            passed=passed,
            score=learning_score,
            message=f"Learning validity score: {learning_score:.2f}",
            details={
                "decision_count": len(decisions),
                "variance": variance,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
            },
            timestamp=datetime.now(),
        )

    @staticmethod
    def _are_contradictory(belief1: str, belief2: str, value1: float, value2: float) -> bool:
        """Check if two beliefs are contradictory (simplified heuristic)."""
        # Simple contradiction detection based on keywords
        contradiction_pairs = [
            ("justice", "mercy"),
            ("individual", "collective"),
            ("freedom", "security"),
            ("tradition", "progress"),
        ]

        belief1_lower = belief1.lower()
        belief2_lower = belief2.lower()

        for word1, word2 in contradiction_pairs:
            if (word1 in belief1_lower and word2 in belief2_lower) or (
                word2 in belief1_lower and word1 in belief2_lower
            ):
                # Check if values are indeed contradictory
                return (value1 > 0 and value2 < 0) or (value1 < 0 and value2 > 0)

        return False


class SocietyValidator:
    """Validates society-level behavior and dynamics."""

    @staticmethod
    def validate_network_integrity(society) -> ValidationResult:
        """Check if social network structure is valid."""
        if not hasattr(society, "social_network"):
            return ValidationResult(
                check_name="network_integrity",
                validation_type=ValidationType.NETWORK_INTEGRITY,
                severity=ValidationSeverity.WARNING,
                passed=False,
                score=0.0,
                message="Society has no social network",
                details={},
                timestamp=datetime.now(),
            )

        G = society.social_network
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        integrity_score = 0.0
        issues = []

        # Check for basic network properties
        if n_nodes > 0:
            integrity_score += 0.2
        else:
            issues.append("No nodes in network")

        # Check connectivity
        if n_nodes > 1:
            try:
                import networkx as nx

                if nx.is_connected(G.to_undirected()):
                    integrity_score += 0.3
                else:
                    # Check largest component size
                    largest_cc_size = len(max(nx.connected_components(G.to_undirected()), key=len))
                    connectivity_ratio = largest_cc_size / n_nodes
                    integrity_score += 0.3 * connectivity_ratio
                    if connectivity_ratio < 0.8:
                        issues.append(f"Network fragmented: {connectivity_ratio:.2f} connectivity")
            except ImportError:
                issues.append("NetworkX not available for connectivity check")

        # Check for reasonable density
        if n_nodes > 1:
            max_edges = n_nodes * (n_nodes - 1) / 2
            density = n_edges / max_edges
            if 0.1 <= density <= 0.9:  # Reasonable range
                integrity_score += 0.3
            else:
                issues.append(f"Unusual network density: {density:.2f}")

        # Check for self-loops (usually undesirable)
        try:
            # For newer NetworkX versions
            self_loops = list(G.nodes_with_selfloops())
        except AttributeError:
            # For older NetworkX versions or manual check
            self_loops = [node for node in G.nodes() if G.has_edge(node, node)]

        if not self_loops:
            integrity_score += 0.2
        else:
            issues.append(f"Found {len(self_loops)} self-loops")

        passed = integrity_score >= 0.7
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="network_integrity",
            validation_type=ValidationType.NETWORK_INTEGRITY,
            severity=severity,
            passed=passed,
            score=integrity_score,
            message=f"Network integrity score: {integrity_score:.2f}",
            details={
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0,
                "issues": issues,
            },
            timestamp=datetime.now(),
        )

    @staticmethod
    def validate_social_stability(society) -> ValidationResult:
        """Check if society exhibits stable social dynamics."""
        if not hasattr(society, "agents") or len(society.agents) < 2:
            return ValidationResult(
                check_name="social_stability",
                validation_type=ValidationType.SOCIETY_STABILITY,
                severity=ValidationSeverity.WARNING,
                passed=False,
                score=0.0,
                message="Insufficient agents for stability analysis",
                details={"agent_count": len(society.agents) if hasattr(society, "agents") else 0},
                timestamp=datetime.now(),
            )

        stability_score = 0.0
        issues = []

        # Check belief distribution
        all_beliefs = []
        for agent in society.agents:
            if hasattr(agent, "beliefs") and agent.beliefs.beliefs:
                all_beliefs.extend(agent.beliefs.beliefs.values())

        if all_beliefs:
            belief_variance = np.var(all_beliefs)
            # Reasonable variance indicates healthy diversity
            if 0.1 <= belief_variance <= 1.0:
                stability_score += 0.4
            else:
                issues.append(f"Unusual belief variance: {belief_variance:.2f}")

        # Check for extreme polarization
        if len(society.agents) >= 2:
            positive_count = sum(
                1
                for agent in society.agents
                if hasattr(agent, "beliefs")
                and agent.beliefs.beliefs
                and np.mean(list(agent.beliefs.beliefs.values())) > 0
            )
            polarization = abs(positive_count / len(society.agents) - 0.5) * 2

            if polarization <= 0.8:  # Not extremely polarized
                stability_score += 0.3
            else:
                issues.append(f"Extreme polarization: {polarization:.2f}")

        # Check social network stability (if available)
        if hasattr(society, "social_network") and society.social_network.number_of_nodes() > 1:
            try:
                import networkx as nx

                G = society.social_network
                # Check for reasonable clustering
                clustering = nx.average_clustering(G)
                if 0.1 <= clustering <= 0.9:
                    stability_score += 0.3
                else:
                    issues.append(f"Unusual clustering coefficient: {clustering:.2f}")
            except ImportError:
                pass

        passed = stability_score >= 0.6
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="social_stability",
            validation_type=ValidationType.SOCIETY_STABILITY,
            severity=severity,
            passed=passed,
            score=stability_score,
            message=f"Social stability score: {stability_score:.2f}",
            details={
                "agent_count": len(society.agents),
                "belief_variance": np.var(all_beliefs) if all_beliefs else 0.0,
                "issues": issues,
            },
            timestamp=datetime.now(),
        )

    @staticmethod
    def validate_scenario_responses(society, scenario) -> ValidationResult:
        """Check if society's response to a scenario is reasonable."""
        if not hasattr(society, "agents") or not society.agents:
            return ValidationResult(
                check_name="scenario_response",
                validation_type=ValidationType.SCENARIO_RESPONSE,
                severity=ValidationSeverity.WARNING,
                passed=False,
                score=0.0,
                message="No agents to respond to scenario",
                details={},
                timestamp=datetime.now(),
            )

        response_score = 0.0
        response_details = {"agent_responses": [], "response_variance": 0.0, "avg_confidence": 0.0}

        # Collect agent responses
        agent_responses = []
        confidences = []

        for i, agent in enumerate(society.agents):
            try:
                # Simulate decision making for the scenario
                if hasattr(agent, "make_decision"):
                    decision = agent.make_decision(scenario)
                    if isinstance(decision, dict):
                        value = decision.get("value", 0.5)
                        confidence = decision.get("confidence", 0.5)
                    else:
                        value = float(decision) if decision is not None else 0.5
                        confidence = 0.5

                    agent_responses.append(value)
                    confidences.append(confidence)
                    response_details["agent_responses"].append(
                        {"agent_id": i, "response": value, "confidence": confidence}
                    )
            except Exception as e:
                warnings.warn(f"Error getting response from agent {i}: {e}")

        if agent_responses:
            # Check response diversity (not all identical)
            response_variance = np.var(agent_responses)
            response_details["response_variance"] = response_variance

            if response_variance > 0.01:  # Some diversity
                response_score += 0.4

            # Check confidence levels (agents should have some confidence)
            avg_confidence = np.mean(confidences)
            response_details["avg_confidence"] = avg_confidence

            if avg_confidence >= 0.3:
                response_score += 0.3

            # Check for reasonable response range
            min_response = min(agent_responses)
            max_response = max(agent_responses)
            if 0.0 <= min_response <= 1.0 and 0.0 <= max_response <= 1.0:
                response_score += 0.3

        passed = response_score >= 0.5
        severity = ValidationSeverity.INFO if passed else ValidationSeverity.WARNING

        return ValidationResult(
            check_name="scenario_response",
            validation_type=ValidationType.SCENARIO_RESPONSE,
            severity=severity,
            passed=passed,
            score=response_score,
            message=f"Scenario response validity: {response_score:.2f}",
            details=response_details,
            timestamp=datetime.now(),
        )


class AnomalyDetector:
    """Detects anomalies and extreme behaviors in simulations."""

    @staticmethod
    def detect_extreme_behaviors(society) -> List[ValidationResult]:
        """Detect agents with extreme or anomalous behaviors."""
        anomalies = []

        if not hasattr(society, "agents") or not society.agents:
            return anomalies

        # Collect all agent decision values
        all_decisions = []
        agent_decision_counts = []

        for agent in society.agents:
            if hasattr(agent, "decision_history") and agent.decision_history:
                agent_decisions = [d.get("value", 0.5) for d in agent.decision_history]
                all_decisions.extend(agent_decisions)
                agent_decision_counts.append(len(agent_decisions))
            else:
                agent_decision_counts.append(0)

        if not all_decisions:
            return anomalies

        # Calculate global statistics
        global_mean = np.mean(all_decisions)
        global_std = np.std(all_decisions)

        # Check each agent for anomalies
        for i, agent in enumerate(society.agents):
            if not hasattr(agent, "decision_history") or not agent.decision_history:
                continue

            agent_decisions = [d.get("value", 0.5) for d in agent.decision_history]
            agent_mean = np.mean(agent_decisions)

            # Check for extreme deviation from global mean
            z_score = abs(agent_mean - global_mean) / (global_std + 1e-6)

            if z_score > 2.5:  # More than 2.5 standard deviations
                anomalies.append(
                    ValidationResult(
                        check_name=f"extreme_behavior_agent_{i}",
                        validation_type=ValidationType.AGENT_CONSISTENCY,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        score=1.0 - min(1.0, z_score / 5.0),
                        message=f"Agent {i} shows extreme behavior (z-score: {z_score:.2f})",
                        details={
                            "agent_id": i,
                            "z_score": z_score,
                            "agent_mean": agent_mean,
                            "global_mean": global_mean,
                            "decision_count": len(agent_decisions),
                        },
                        timestamp=datetime.now(),
                    )
                )

        return anomalies

    @staticmethod
    def detect_network_anomalies(society) -> List[ValidationResult]:
        """Detect anomalies in social network structure."""
        anomalies = []

        if not hasattr(society, "social_network"):
            return anomalies

        G = society.social_network
        if G.number_of_nodes() < 2:
            return anomalies

        try:
            import networkx as nx

            # Check for isolated nodes
            isolated_nodes = list(nx.isolates(G))
            if isolated_nodes and len(isolated_nodes) > G.number_of_nodes() * 0.1:
                anomalies.append(
                    ValidationResult(
                        check_name="isolated_nodes",
                        validation_type=ValidationType.NETWORK_INTEGRITY,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        score=1.0 - len(isolated_nodes) / G.number_of_nodes(),
                        message=f"High number of isolated nodes: {len(isolated_nodes)}",
                        details={"isolated_nodes": isolated_nodes},
                        timestamp=datetime.now(),
                    )
                )

            # Check for nodes with extremely high degree
            degrees = dict(G.degree())
            if degrees:
                max_degree = max(degrees.values())
                avg_degree = np.mean(list(degrees.values()))

                if max_degree > avg_degree * 3:  # More than 3x average
                    high_degree_nodes = [
                        node for node, degree in degrees.items() if degree > avg_degree * 2
                    ]
                    anomalies.append(
                        ValidationResult(
                            check_name="high_degree_nodes",
                            validation_type=ValidationType.NETWORK_INTEGRITY,
                            severity=ValidationSeverity.INFO,
                            passed=True,
                            score=0.8,
                            message=f"Nodes with unusually high degree: {len(high_degree_nodes)}",
                            details={
                                "high_degree_nodes": high_degree_nodes,
                                "max_degree": max_degree,
                                "avg_degree": avg_degree,
                            },
                            timestamp=datetime.now(),
                        )
                    )

        except ImportError:
            pass

        return anomalies


class ValidationSuite:
    """Main validation orchestrator."""

    def __init__(self):
        self.agent_validator = AgentValidator()
        self.society_validator = SocietyValidator()
        self.anomaly_detector = AnomalyDetector()
        self.validation_history: List[ValidationResult] = []

    def validate_agent(self, agent, agent_id: str = None) -> List[ValidationResult]:
        """Run all validation checks on a single agent."""
        results = []

        results.append(self.agent_validator.validate_belief_consistency(agent))
        results.append(self.agent_validator.validate_decision_plausibility(agent))
        results.append(self.agent_validator.validate_learning_progress(agent))

        # Store in history
        self.validation_history.extend(results)

        return results

    def validate_society(self, society, scenario=None) -> List[ValidationResult]:
        """Run all validation checks on a society."""
        results = []

        results.append(self.society_validator.validate_network_integrity(society))
        results.append(self.society_validator.validate_social_stability(society))

        if scenario:
            results.append(self.society_validator.validate_scenario_responses(society, scenario))

        # Detect anomalies
        results.extend(self.anomaly_detector.detect_extreme_behaviors(society))
        results.extend(self.anomaly_detector.detect_network_anomalies(society))

        # Store in history
        self.validation_history.extend(results)

        return results

    def validate_all(self, society, scenario=None) -> Dict[str, List[ValidationResult]]:
        """Run comprehensive validation on society and all agents."""
        all_results = {"society": [], "agents": {}, "anomalies": []}

        # Validate society
        society_results = self.validate_society(society, scenario)
        all_results["society"] = society_results

        # Validate individual agents
        if hasattr(society, "agents"):
            for i, agent in enumerate(society.agents):
                agent_id = f"agent_{i}"
                agent_results = self.validate_agent(agent, agent_id)
                all_results["agents"][agent_id] = agent_results

        return all_results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Generate summary of validation results."""
        if not self.validation_history:
            return {"error": "No validation results available"}

        summary = {
            "total_checks": len(self.validation_history),
            "passed_checks": sum(1 for r in self.validation_history if r.passed),
            "failed_checks": sum(1 for r in self.validation_history if not r.passed),
            "by_severity": {},
            "by_type": {},
            "avg_score": np.mean([r.score for r in self.validation_history]),
            "critical_issues": [],
        }

        # Group by severity
        for severity in ValidationSeverity:
            count = sum(1 for r in self.validation_history if r.severity == severity)
            summary["by_severity"][severity.value] = count

        # Group by type
        for val_type in ValidationType:
            type_results = [r for r in self.validation_history if r.validation_type == val_type]
            summary["by_type"][val_type.value] = {
                "count": len(type_results),
                "passed": sum(1 for r in type_results if r.passed),
                "avg_score": np.mean([r.score for r in type_results]) if type_results else 0.0,
            }

        # Critical issues
        critical_issues = [
            r
            for r in self.validation_history
            if r.severity == ValidationSeverity.CRITICAL
            or (r.severity == ValidationSeverity.ERROR and not r.passed)
        ]
        summary["critical_issues"] = [
            {"check": r.check_name, "message": r.message, "score": r.score} for r in critical_issues
        ]

        return summary

    def export_validation_results(self, filepath: str, format: str = "json") -> None:
        """Export validation results to file."""
        export_data = []
        for result in self.validation_history:
            export_data.append(
                {
                    "check_name": result.check_name,
                    "validation_type": result.validation_type.value,
                    "severity": result.severity.value,
                    "passed": result.passed,
                    "score": result.score,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat(),
                }
            )

        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            import csv

            if export_data:
                fieldnames = [
                    "check_name",
                    "validation_type",
                    "severity",
                    "passed",
                    "score",
                    "message",
                    "timestamp",
                ]

                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in export_data:
                        csv_row = {k: v for k, v in row.items() if k in fieldnames}
                        writer.writerow(csv_row)

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
