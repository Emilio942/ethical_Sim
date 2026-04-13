"""
Comprehensive metrics and evaluation system for ethical agent simulations.

This module provides tools to measure ethical consistency, social dynamics,
and decision quality across agents and societies.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, cast
from dataclasses import dataclass
from enum import Enum
import json
import csv
from pathlib import Path
from datetime import datetime


class MetricType(Enum):
    """Types of metrics available in the system."""

    ETHICAL_CONSISTENCY = "ethical_consistency"
    SOCIAL_DYNAMICS = "social_dynamics"
    DECISION_QUALITY = "decision_quality"
    LEARNING_PROGRESS = "learning_progress"
    POLARIZATION = "polarization"
    CONSENSUS = "consensus"


@dataclass
class MetricResult:
    """Result of a metric calculation."""

    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    description: str


class EthicalConsistencyMetrics:
    """Measures ethical consistency within and across agents."""

    @staticmethod
    def internal_consistency(agent) -> float:
        """Measure internal consistency of an agent's beliefs."""
        if not hasattr(agent, "beliefs") or not agent.beliefs:
            return 0.0

        beliefs = agent.beliefs
        belief_values = [b.strength for b in beliefs.values()]

        if len(belief_values) < 2:
            return 1.0

        # Calculate variance in belief strengths
        variance = np.var(belief_values)
        # Convert to consistency score (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance)

        return float(min(1.0, max(0.0, consistency)))

    @staticmethod
    def temporal_stability(agent, history_length: int = 10) -> float:
        """Measure how stable an agent's beliefs are over time."""
        if not hasattr(agent, "decision_history") or len(agent.decision_history) < 2:
            return 1.0

        recent_decisions = agent.decision_history[-history_length:]
        if len(recent_decisions) < 2:
            return 1.0

        # Calculate consistency of decisions over time
        decision_values = [d.get("value", 0.5) for d in recent_decisions]
        if len(set(decision_values)) == 1:
            return 1.0

        # Calculate standard deviation and convert to stability score
        std_dev = np.std(decision_values)
        stability = 1.0 / (1.0 + std_dev)

        return float(min(1.0, max(0.0, stability)))

    @staticmethod
    def cross_domain_consistency(agent, domains: Optional[List[str]] = None) -> float:
        """Measure consistency across different ethical domains."""
        if not hasattr(agent, "beliefs") or not agent.beliefs:
            return 0.0

        if domains is None:
            domains = ["justice", "care", "loyalty", "authority", "purity"]

        domain_scores = {}
        for domain in domains:
            domain_beliefs = {
                k: v.strength for k, v in agent.beliefs.items() if domain.lower() in k.lower()
            }
            if domain_beliefs:
                domain_scores[domain] = np.mean(list(domain_beliefs.values()))

        if len(domain_scores) < 2:
            return 1.0

        # Calculate variance across domains
        scores = list(domain_scores.values())
        variance = np.var(scores)
        consistency = 1.0 / (1.0 + variance)

        return float(min(1.0, max(0.0, consistency)))


class SocialDynamicsMetrics:
    """Measures social dynamics and group behavior."""

    @staticmethod
    def polarization_index(society) -> float:
        """
        Measure polarization using Jensen-Shannon Divergence (JSD).
        Replaces simple Euclidean distance with Information Topology.
        """
        if not hasattr(society, "agents") or len(society.agents) < 2:
            return 0.0

        def _jsd(p, q):
            p = np.clip(p, 1e-12, 1.0)
            q = np.clip(q, 1e-12, 1.0)
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log2(p/m)) + np.sum(q * np.log2(q/m)))

        agents_list = list(society.agents.values())
        jsd_values = []
        
        for i, a1 in enumerate(agents_list):
            for a2 in agents_list[i+1:]:
                # Extract belief probability distributions
                b1 = np.array([b.strength for b in a1.beliefs.values()])
                b2 = np.array([b.strength for b in a2.beliefs.values()])
                # Normalize to distributions
                p = b1 / (np.sum(b1) + 1e-8)
                q = b2 / (np.sum(b2) + 1e-8)
                jsd_values.append(np.sqrt(_jsd(p, q)))

        return float(np.mean(jsd_values)) if jsd_values else 0.0

    @staticmethod
    def wasserstein_alignment(society) -> float:
        """
        Computes the Wasserstein-1 (Optimal Transport) distance between agents' beliefs.
        Provides a geometric measure of alignment considering the 'cost' of belief shift.
        """
        from scipy.stats import wasserstein_distance
        if not hasattr(society, "agents") or len(society.agents) < 2:
            return 0.0

        agents_list = list(society.agents.values())
        w1_values = []
        
        for i, a1 in enumerate(agents_list):
            for a2 in agents_list[i+1:]:
                b1 = np.array([b.strength for b in a1.beliefs.values()])
                b2 = np.array([b.strength for b in a2.beliefs.values()])
                # Wasserstein-1 distance
                w1_values.append(wasserstein_distance(b1, b2))

        return float(np.mean(w1_values)) if w1_values else 0.0

    @staticmethod
    def estimate_consensus_speed(society) -> float:
        """
        Estimates the 'speed of consensus' based on the Ollivier-Ricci Curvature 
        of the social graph. Speed is bounded by 1/kappa.
        """
        if not hasattr(society, "social_network") or society.social_network.number_of_nodes() < 2:
            return 0.0

        # Coarse approximation of Ollivier-Ricci Curvature (kappa)
        # kappa(u,v) = 1 - W1(m_u, m_v) / d(u,v)
        # For simplicity, we use the graph density and clustering as proxies for curvature
        density = nx.density(society.social_network)
        clustering = nx.average_clustering(society.social_network)
        
        # Heuristic: kappa approx proportional to clustering and density
        kappa = (density + clustering) / 2.0
        
        # Consensus speed bound: Rate of convergence is (1 - kappa)^t
        # We return the contraction factor kappa
        return float(kappa)

    @staticmethod
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fisher-Rao distance between two belief distributions.
        d = arccos(sum(sqrt(p_i * q_i))). Represented as an angle on the sphere.
        """
        # Normalize
        p = p / (np.sum(p) + 1e-8)
        q = q / (np.sum(q) + 1e-8)
        
        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0.0, 1.0)
        
        return float(np.arccos(bc))

    @staticmethod
    def calculate_moral_volatility(belief_history: np.ndarray, dt: float = 1.0) -> float:
        """
        Derives the 'Moral Volatility' using Ito's Lemma logic.
        Measures the quadratic variation of the belief process.
        """
        if len(belief_history) < 2:
            return 0.0
            
        # Diffusion coefficient sigma estimated from increments
        diffs = np.diff(belief_history)
        sigma_sq = np.var(diffs) / dt
        
        # In a real model, we would use f'(b)^2 * sigma^2
        # Here we return the raw diffusion variance as volatility
        return float(np.sqrt(sigma_sq))

    @staticmethod
    def detect_cohomological_obstruction(society) -> float:
        """
        Detects global consensus obstructions using Sheaf Cohomology (H^1).
        A non-zero return value indicates a 'Moral Twist' (paradoxical loop).
        """
        if not hasattr(society, "social_network") or society.social_network.number_of_nodes() < 3:
            return 0.0

        # Find cycles in the social graph (Homology H1)
        cycles = nx.cycle_basis(society.social_network)
        obstruction_count = 0
        
        for cycle in cycles:
            # Calculate the 'Net Twist' around the cycle
            net_twist = 0.0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                
                if u in society.agents and v in society.agents:
                    b_u = np.array([b.strength for b in society.agents[u].beliefs.values()])
                    b_v = np.array([b.strength for b in society.agents[v].beliefs.values()])
                    
                    # Gradient of belief shift
                    net_twist += np.linalg.norm(b_u - b_v)
            
            # If net twist is large, the cycle is 'knotted' (obstruction)
            if net_twist > 1.5: 
                obstruction_count += 1
                
        return float(obstruction_count)

    # --- MATHEMATISCHE ANFRAGE: THERMODYNAMIK ---
    # Frage: Wenn wir die Gesellschaft als offenes thermodynamisches System 
    # betrachten, kann eine statistische Gleichung die 'Arbeit' berechnen, 
    # die für den Übergang zwischen moralischen Gleichgewichten nötig ist?

    # --- MATHEMATISCHE ANFRAGE: TOPOLOGISCHE TRAJEKTORIEN ---
    # Frage: Können ethische Übergänge zwischen moralischen Gleichgewichten 
    # mittels topologischer Invarianten auf dem symplektischen Manifold 
    # quantifiziert werden?

    # --- MATHEMATISCHE ANFRAGE: INFORMATIONSTHEORIE ---
    # Frage: Was ist die minimale Anzahl an 'Moralischen Bits', um ein 
    # Rechtssystem ohne Integritätsverlust darzustellen (Rate-Distortion)?

    # --- MATHEMATISCHE ANFRAGE: FUNKTIONALANALYSIS ---
    # Frage: Erlaubt der Spektralsatz, die gesamte Gesellschaft in 
    # unabhängige 'Moralische Eigenmoden' zu zerlegen?

    @staticmethod
    def graph_belief_denoising(society, cutoff_lambda: float = 0.5) -> Dict[str, float]:
        """
        Applies Graph Fourier Transform (GFT) to belief signals.
        Uses a Graph Spectral Filter to 'de-noise' conflict (high-frequency) 
        and preserve consensus (low-frequency).
        """
        if not hasattr(society, "social_network") or society.social_network.number_of_nodes() < 2:
            return {}

        # 1. Compute Laplacian
        # Ensure we only use nodes that are actually agents
        agent_ids = list(society.agents.keys())
        subgraph = society.social_network.subgraph(agent_ids)
        L = nx.laplacian_matrix(subgraph).toarray()
        eigvals, U = np.linalg.eigh(L)
        
        # 2. Extract belief signal x
        x = np.array([np.mean([b.strength for b in society.agents[aid].beliefs.values()]) for aid in agent_ids])
        
        # 3. GFT: x_hat = U^T * x
        x_hat = np.dot(U.T, x)
        
        # 4. Low-pass Filter H(lambda)
        H = np.where(eigvals <= cutoff_lambda, 1.0, 0.0)
        
        # 5. Inverse GFT: x_tilde = U * H * x_hat
        x_denoised = np.dot(U, H * x_hat)
        
        return {aid: float(val) for aid, val in zip(agent_ids, x_denoised)}

    @staticmethod
    def consensus_level(society) -> float:
        """
        Measure level of consensus using average Fisher-Rao distance.
        High consensus = Low average distance between worldviews.
        """
        if not hasattr(society, "agents") or len(society.agents) < 2:
            return 1.0

        from analysis.metrics import SocialDynamicsMetrics
        agents_list = list(society.agents.values())
        distances = []
        
        for i, a1 in enumerate(agents_list):
            for a2 in agents_list[i+1:]:
                p = np.array([b.strength for b in a1.beliefs.values()])
                q = np.array([b.strength for b in a2.beliefs.values()])
                # Inverse of Fisher-Rao distance as consensus measure
                d = SocialDynamicsMetrics.fisher_rao_distance(p, q)
                distances.append(d)

        # Map average distance to 0-1 scale (pi/2 is max distance)
        avg_dist = np.mean(distances) if distances else 0.0
        consensus = 1.0 - (avg_dist / (np.pi / 2))
        return float(np.clip(consensus, 0.0, 1.0))

    @staticmethod
    def calculate_ethical_work(initial_beliefs: np.ndarray, final_beliefs: np.ndarray, temp: float = 1.0) -> float:
        """
        Calculates the 'Work' required for a moral transition.
        META-FIX: Implements Adèlic volume weighting (dV_i) to prevent 
        abstraction leakage in non-Euclidean manifolds.
        """
        # Volume factor dV (invariant measure on the belief manifold)
        # For a sphere, dV depends on the belief intensity (det g)
        dv_i = np.sqrt(np.abs(initial_beliefs) + 1e-8)
        dv_f = np.sqrt(np.abs(final_beliefs) + 1e-8)
        
        # Energy E = -log(belief_strength + eps)
        e_i = -np.log(initial_beliefs + 1e-8)
        e_f = -np.log(final_beliefs + 1e-8)
        
        # Weighted Partition Functions Z
        z_i = np.mean(np.exp(-e_i / temp) * dv_i)
        z_f = np.mean(np.exp(-e_f / temp) * dv_f)
        
        delta_f = -temp * np.log(z_f / (z_i + 1e-8))
        return float(delta_f)

    @staticmethod
    def detect_cohomological_obstruction(society) -> float:
        """
        Detects global consensus obstructions using Sheaf Cohomology (H^1).
        A non-zero return value indicates a 'Moral Twist' (paradoxical loop).
        """
        if not hasattr(society, "social_network") or society.social_network.number_of_nodes() < 3:
            return 0.0

        # Find cycles in the social graph (Homology H1)
        import networkx as nx
        cycles = nx.cycle_basis(society.social_network)
        obstruction_count = 0
        
        for cycle in cycles:
            net_twist = 0.0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                
                if u in society.agents and v in society.agents:
                    b_u = np.array([b.strength for b in society.agents[u].beliefs.values()])
                    b_v = np.array([b.strength for b in society.agents[v].beliefs.values()])
                    net_twist += np.linalg.norm(b_u - b_v)
            
            if net_twist > 1.5: 
                obstruction_count += 1
                
        return float(obstruction_count)

    @staticmethod
    def calculate_ethical_entropy(society) -> float:
        """
        Calculates the Shannon Entropy of the society's moral distribution.
        Measures the degree of 'Moral Disorder'.
        """
        if not hasattr(society, "agents") or not society.agents:
            return 0.0
            
        # Collect flattened belief distributions
        all_strengths = []
        for agent in society.agents.values():
            all_strengths.extend([b.strength for b in agent.beliefs.values()])
            
        if not all_strengths:
            return 0.0
            
        # Discretize into K bins
        counts, _ = np.histogram(all_strengths, bins=10, range=(0, 1))
        probs = counts / (np.sum(counts) + 1e-8)
        probs = np.clip(probs, 1e-12, 1.0)
        
        return -float(np.sum(probs * np.log2(probs)))

    @staticmethod
    def padic_moral_distance(belief_a: np.ndarray, belief_b: np.ndarray, p: int = 2) -> float:
        """
        Computes the p-adic (non-Archimedean) distance between two beliefs.
        Proximity depends on shared hierarchical roots (LCA logic).
        |x - y|_p = p^(-k) where k is the depth of the first differing digit.
        """
        # Convert to discrete addresses (simplified tree depth)
        depth = 8
        scale = p ** depth
        
        val_a = int(np.mean(belief_a) * scale)
        val_b = int(np.mean(belief_b) * scale)
        
        if val_a == val_b:
            return 0.0
            
        # Find the smallest k such that (val_a - val_b) is divisible by p^k
        diff = abs(val_a - val_b)
        k = 0
        while diff % (p ** (k + 1)) == 0 and k < depth:
            k += 1
            
        return float(p ** (-k))
        if not hasattr(society, "agents") or len(society.agents) < 2:
            return 1.0

        # Count pairs with similar beliefs
        consensus_pairs = 0
        total_pairs = 0

        agents_list = list(society.agents.values())
        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i + 1 :]:
                if hasattr(agent1, "beliefs") and hasattr(agent2, "beliefs"):
                    distance = SocialDynamicsMetrics._belief_distance(agent1, agent2)
                    if distance <= threshold:
                        consensus_pairs += 1
                    total_pairs += 1

        if total_pairs == 0:
            return 1.0

        return consensus_pairs / total_pairs

    @staticmethod
    def network_cohesion(society) -> float:
        """Measure network cohesion based on social connections."""
        if not hasattr(society, "social_network") or society.social_network.number_of_nodes() < 2:
            return 0.0

        # Calculate network density
        G = society.social_network
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2

        if max_edges == 0:
            return 0.0

        density = n_edges / max_edges
        return float(min(1.0, density))

    @staticmethod
    def influence_distribution(society) -> Dict[str, float]:
        """Measure distribution of influence among agents."""
        if not hasattr(society, "social_network"):
            return {}

        G = society.social_network
        if G.number_of_nodes() == 0:
            return {}

        # Calculate centrality measures
        centrality_measures = {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G),
            "closeness": nx.closeness_centrality(G),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06),
        }

        # Calculate Gini coefficient for each measure
        gini_scores = {}
        for measure_name, centralities in centrality_measures.items():
            values = list(centralities.values())
            if values:
                gini_scores[f"{measure_name}_gini"] = SocialDynamicsMetrics._gini_coefficient(
                    values
                )

        return gini_scores

    @staticmethod
    def _belief_distance(agent1, agent2) -> float:
        """Calculate distance between two agents' beliefs."""
        if not (hasattr(agent1, "beliefs") and hasattr(agent2, "beliefs")):
            return 1.0

        beliefs1 = agent1.beliefs  # Direct dictionary access
        beliefs2 = agent2.beliefs  # Direct dictionary access

        if not beliefs1 or not beliefs2:
            return 1.0

        # Find common beliefs by name
        common_beliefs = set(beliefs1.keys()) & set(beliefs2.keys())
        if not common_beliefs:
            return 1.0

        # Calculate Euclidean distance using belief strengths
        distance = 0.0
        for belief_name in common_beliefs:
            belief1 = beliefs1[belief_name]
            belief2 = beliefs2[belief_name]
            # Get strength from belief objects
            strength1 = belief1.strength if hasattr(belief1, "strength") else 0.5
            strength2 = belief2.strength if hasattr(belief2, "strength") else 0.5
            distance += (strength1 - strength2) ** 2

        return float(np.sqrt(distance / len(common_beliefs)))

    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or len(values) < 2:
            return 0.0

        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)

    # --- MATHEMATICAL INQUIRY 2: INFORMATION TOPOLOGY ---
    # Question: Could the 'polarization_index' be redefined using Jensen-Shannon Divergence (JSD) 
    # over the probability distributions of beliefs? Furthermore, if we model the social network 
    # as a k-dimensional Simplicial Complex, how would the Betti numbers (β_k) of the 
    # persistent homology track the topological "holes" in ethical consensus?


class DecisionQualityMetrics:
    """Measures quality of decisions made by agents."""

    @staticmethod
    def decision_confidence(agent) -> float:
        """Measure confidence in recent decisions."""
        if not hasattr(agent, "decision_history") or not agent.decision_history:
            return 0.0

        recent_decisions = agent.decision_history[-10:]
        confidences = [d.get("confidence", 0.5) for d in recent_decisions]

        return np.mean(confidences) if confidences else 0.0

    @staticmethod
    def decision_diversity(agent, window_size: int = 10) -> float:
        """Measure diversity in decision-making patterns."""
        if not hasattr(agent, "decision_history") or len(agent.decision_history) < 2:
            return 0.0

        recent_decisions = agent.decision_history[-window_size:]
        decision_values = [d.get("value", 0.5) for d in recent_decisions]

        if len(set(decision_values)) == 1:
            return 0.0

        # Calculate entropy as diversity measure
        unique_values, counts = np.unique(decision_values, return_counts=True)
        probabilities = counts / len(decision_values)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(decision_values))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def ethical_alignment(agent, target_principles: Dict[str, float]) -> float:
        """Measure how well agent's decisions align with target ethical principles."""
        if not hasattr(agent, "beliefs") or not agent.beliefs.beliefs:
            return 0.0

        alignment_scores = []
        for principle, target_value in target_principles.items():
            if principle in agent.beliefs.beliefs:
                agent_value = agent.beliefs.beliefs[principle]
                # Calculate alignment (1 - normalized absolute difference)
                alignment = 1.0 - abs(agent_value - target_value) / 2.0
                alignment_scores.append(alignment)

        return float(np.mean(alignment_scores)) if alignment_scores else 0.0


class LearningProgressMetrics:
    """Measures learning and adaptation progress."""

    @staticmethod
    def learning_rate(agent) -> float:
        """Measure how quickly an agent is learning/adapting."""
        if not hasattr(agent, "decision_history") or len(agent.decision_history) < 5:
            return 0.0

        # Calculate change in decision patterns over time
        recent_decisions = agent.decision_history[-10:]
        older_decisions = (
            agent.decision_history[-20:-10] if len(agent.decision_history) >= 20 else []
        )

        if not older_decisions:
            return 0.0

        recent_avg = np.mean([d.get("value", 0.5) for d in recent_decisions])
        older_avg = np.mean([d.get("value", 0.5) for d in older_decisions])

        # Learning rate as absolute change
        return float(abs(recent_avg - older_avg))

    @staticmethod
    def adaptation_efficiency(agent) -> float:
        """Measure how efficiently an agent adapts to new scenarios."""
        if not hasattr(agent, "decision_history") or len(agent.decision_history) < 3:
            return 0.0

        # Look for patterns in decision improvement
        decisions = agent.decision_history[-15:]
        if len(decisions) < 3:
            return 0.0

        # Calculate trend in decision confidence
        confidences = [d.get("confidence", 0.5) for d in decisions]
        if len(confidences) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(confidences))
        slope = np.polyfit(x, confidences, 1)[0]

        # Normalize slope to 0-1 range
        return float(min(1.0, max(0.0, slope + 0.5)))


class MetricsCollector:
    """Main class for collecting and managing metrics."""

    def __init__(self):
        self.metric_history: List[MetricResult] = []
        self.ethical_metrics = EthicalConsistencyMetrics()
        self.social_metrics = SocialDynamicsMetrics()
        self.decision_metrics = DecisionQualityMetrics()
        self.learning_metrics = LearningProgressMetrics()

    def collect_agent_metrics(self, agent, agent_id: Optional[str] = None) -> Dict[str, MetricResult]:
        """Collect all metrics for a single agent."""
        agent_id = agent_id or f"agent_{id(agent)}"
        metrics = {}

        # Ethical consistency metrics
        metrics["internal_consistency"] = MetricResult(
            name=f"{agent_id}_internal_consistency",
            value=self.ethical_metrics.internal_consistency(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Internal consistency of agent's beliefs",
        )

        metrics["temporal_stability"] = MetricResult(
            name=f"{agent_id}_temporal_stability",
            value=self.ethical_metrics.temporal_stability(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Stability of agent's beliefs over time",
        )

        metrics["cross_domain_consistency"] = MetricResult(
            name=f"{agent_id}_cross_domain_consistency",
            value=self.ethical_metrics.cross_domain_consistency(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Consistency across ethical domains",
        )

        # Decision quality metrics
        metrics["decision_confidence"] = MetricResult(
            name=f"{agent_id}_decision_confidence",
            value=self.decision_metrics.decision_confidence(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Confidence in recent decisions",
        )

        metrics["decision_diversity"] = MetricResult(
            name=f"{agent_id}_decision_diversity",
            value=self.decision_metrics.decision_diversity(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Diversity in decision-making patterns",
        )

        # Learning progress metrics
        metrics["learning_rate"] = MetricResult(
            name=f"{agent_id}_learning_rate",
            value=self.learning_metrics.learning_rate(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Rate of learning and adaptation",
        )

        metrics["adaptation_efficiency"] = MetricResult(
            name=f"{agent_id}_adaptation_efficiency",
            value=self.learning_metrics.adaptation_efficiency(agent),
            timestamp=datetime.now(),
            metadata={"agent_id": agent_id},
            description="Efficiency of adaptation to new scenarios",
        )

        # Store in history
        self.metric_history.extend(metrics.values())

        return metrics

    def collect_society_metrics(self, society) -> Dict[str, MetricResult]:
        """Collect all metrics for a society."""
        metrics = {}

        # Social dynamics metrics
        metrics["polarization"] = MetricResult(
            name="society_polarization",
            value=self.social_metrics.polarization_index(society),
            timestamp=datetime.now(),
            metadata={"n_agents": len(society.agents) if hasattr(society, "agents") else 0},
            description="Level of polarization in society",
        )

        metrics["consensus"] = MetricResult(
            name="society_consensus",
            value=self.social_metrics.consensus_level(society),
            timestamp=datetime.now(),
            metadata={"n_agents": len(society.agents) if hasattr(society, "agents") else 0},
            description="Level of consensus in society",
        )

        metrics["network_cohesion"] = MetricResult(
            name="society_network_cohesion",
            value=self.social_metrics.network_cohesion(society),
            timestamp=datetime.now(),
            metadata={"n_agents": len(society.agents) if hasattr(society, "agents") else 0},
            description="Cohesion of social network",
        )

        # Influence distribution
        influence_dist = self.social_metrics.influence_distribution(society)
        for measure, value in influence_dist.items():
            metrics[f"influence_{measure}"] = MetricResult(
                name=f"society_influence_{measure}",
                value=value,
                timestamp=datetime.now(),
                metadata={"measure_type": measure},
                description=f"Influence distribution: {measure}",
            )

        # Store in history
        self.metric_history.extend(metrics.values())

        return metrics

    def collect_all_metrics(self, society) -> Dict[str, Any]:
        """Collect all metrics for society and its agents."""
        all_metrics: Dict[str, Any] = {}

        # Society metrics
        society_metrics = self.collect_society_metrics(society)
        all_metrics["society"] = society_metrics

        # Agent metrics
        all_metrics["agents"] = {}
        if hasattr(society, "agents"):
            for i, agent in enumerate(society.agents):
                agent_id = f"agent_{i}"
                agent_metrics = self.collect_agent_metrics(agent, agent_id)
                all_metrics["agents"][agent_id] = agent_metrics

        return all_metrics

    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export collected metrics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for export
        export_data = []
        for metric in self.metric_history:
            export_data.append(
                {
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "metadata": metric.metadata,
                    "description": metric.description,
                }
            )

        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format.lower() == "csv":
            if export_data:
                metadata = cast(Dict[str, Any], export_data[0]["metadata"])
                fieldnames = ["name", "value", "timestamp", "description"] + list(
                    metadata.keys()
                )

                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for row in export_data:
                        flat_row = {
                            "name": row["name"],
                            "value": row["value"],
                            "timestamp": row["timestamp"],
                            "description": row["description"],
                        }
                        flat_row.update(cast(Dict[str, Any], row["metadata"]))
                        writer.writerow(flat_row)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all collected metrics."""
        if not self.metric_history:
            return {"error": "No metrics collected yet"}

        # Group metrics by type
        metric_groups: Dict[str, List[float]] = {}
        for metric in self.metric_history:
            metric_type = metric.name.split("_")[1] if "_" in metric.name else "other"
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(metric.value)

        # Calculate summary statistics
        summary: Dict[str, Any] = {
            "total_metrics": len(self.metric_history),
            "collection_period": {
                "start": min(m.timestamp for m in self.metric_history).isoformat(),
                "end": max(m.timestamp for m in self.metric_history).isoformat(),
            },
            "metric_types": {},
        }

        for metric_type, values in metric_groups.items():
            summary["metric_types"][metric_type] = {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
            }

        return summary

    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metric_history.clear()
