"""
Comprehensive metrics and evaluation system for ethical agent simulations.

This module provides tools to measure ethical consistency, social dynamics,
and decision quality across agents and societies.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
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
        if not hasattr(agent, 'beliefs') or not agent.beliefs.beliefs:
            return 0.0
        
        beliefs = agent.beliefs.beliefs
        belief_values = [abs(beliefs[belief]) for belief in beliefs]
        
        if len(belief_values) < 2:
            return 1.0
        
        # Calculate variance in belief strengths
        variance = np.var(belief_values)
        # Convert to consistency score (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + variance)
        
        return min(1.0, max(0.0, consistency))
    
    @staticmethod
    def temporal_stability(agent, history_length: int = 10) -> float:
        """Measure how stable an agent's beliefs are over time."""
        if not hasattr(agent, 'decision_history') or len(agent.decision_history) < 2:
            return 1.0
        
        recent_decisions = agent.decision_history[-history_length:]
        if len(recent_decisions) < 2:
            return 1.0
        
        # Calculate consistency of decisions over time
        decision_values = [d.get('value', 0.5) for d in recent_decisions]
        if len(set(decision_values)) == 1:
            return 1.0
        
        # Calculate standard deviation and convert to stability score
        std_dev = np.std(decision_values)
        stability = 1.0 / (1.0 + std_dev)
        
        return min(1.0, max(0.0, stability))
    
    @staticmethod
    def cross_domain_consistency(agent, domains: List[str] = None) -> float:
        """Measure consistency across different ethical domains."""
        if not hasattr(agent, 'beliefs') or not agent.beliefs.beliefs:
            return 0.0
        
        if domains is None:
            domains = ["justice", "care", "loyalty", "authority", "purity"]
        
        domain_scores = {}
        for domain in domains:
            domain_beliefs = {k: v for k, v in agent.beliefs.beliefs.items() 
                            if domain.lower() in k.lower()}
            if domain_beliefs:
                domain_scores[domain] = np.mean(list(domain_beliefs.values()))
        
        if len(domain_scores) < 2:
            return 1.0
        
        # Calculate variance across domains
        scores = list(domain_scores.values())
        variance = np.var(scores)
        consistency = 1.0 / (1.0 + variance)
        
        return min(1.0, max(0.0, consistency))

class SocialDynamicsMetrics:
    """Measures social dynamics and group behavior."""
    
    @staticmethod
    def polarization_index(society) -> float:
        """Measure polarization in the society."""
        if not hasattr(society, 'agents') or len(society.agents) < 2:
            return 0.0
        
        # Calculate average belief differences between agents
        total_distance = 0.0
        comparisons = 0
        
        agents_list = list(society.agents.values())
        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i+1:]:
                if hasattr(agent1, 'beliefs') and hasattr(agent2, 'beliefs'):
                    distance = SocialDynamicsMetrics._belief_distance(agent1, agent2)
                    total_distance += distance
                    comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        average_distance = total_distance / comparisons
        # Normalize to 0-1 scale
        return min(1.0, average_distance / 2.0)
    
    @staticmethod
    def consensus_level(society, threshold: float = 0.1) -> float:
        """Measure level of consensus in the society."""
        if not hasattr(society, 'agents') or len(society.agents) < 2:
            return 1.0
        
        # Count pairs with similar beliefs
        consensus_pairs = 0
        total_pairs = 0
        
        agents_list = list(society.agents.values())
        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i+1:]:
                if hasattr(agent1, 'beliefs') and hasattr(agent2, 'beliefs'):
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
        if not hasattr(society, 'social_network') or society.social_network.number_of_nodes() < 2:
            return 0.0
        
        # Calculate network density
        G = society.social_network
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) / 2
        
        if max_edges == 0:
            return 0.0
        
        density = n_edges / max_edges
        return min(1.0, density)
    
    @staticmethod
    def influence_distribution(society) -> Dict[str, float]:
        """Measure distribution of influence among agents."""
        if not hasattr(society, 'social_network'):
            return {}
        
        G = society.social_network
        if G.number_of_nodes() == 0:
            return {}
        
        # Calculate centrality measures
        centrality_measures = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        }
        
        # Calculate Gini coefficient for each measure
        gini_scores = {}
        for measure_name, centralities in centrality_measures.items():
            values = list(centralities.values())
            if values:
                gini_scores[f'{measure_name}_gini'] = SocialDynamicsMetrics._gini_coefficient(values)
        
        return gini_scores
    
    @staticmethod
    def _belief_distance(agent1, agent2) -> float:
        """Calculate distance between two agents' beliefs."""
        if not (hasattr(agent1, 'beliefs') and hasattr(agent2, 'beliefs')):
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
            strength1 = belief1.strength if hasattr(belief1, 'strength') else 0.5
            strength2 = belief2.strength if hasattr(belief2, 'strength') else 0.5
            distance += (strength1 - strength2) ** 2
        
        return np.sqrt(distance / len(common_beliefs))
    
    @staticmethod
    def _gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or len(values) < 2:
            return 0.0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

class DecisionQualityMetrics:
    """Measures quality of decisions made by agents."""
    
    @staticmethod
    def decision_confidence(agent) -> float:
        """Measure confidence in recent decisions."""
        if not hasattr(agent, 'decision_history') or not agent.decision_history:
            return 0.0
        
        recent_decisions = agent.decision_history[-10:]
        confidences = [d.get('confidence', 0.5) for d in recent_decisions]
        
        return np.mean(confidences) if confidences else 0.0
    
    @staticmethod
    def decision_diversity(agent, window_size: int = 10) -> float:
        """Measure diversity in decision-making patterns."""
        if not hasattr(agent, 'decision_history') or len(agent.decision_history) < 2:
            return 0.0
        
        recent_decisions = agent.decision_history[-window_size:]
        decision_values = [d.get('value', 0.5) for d in recent_decisions]
        
        if len(set(decision_values)) == 1:
            return 0.0
        
        # Calculate entropy as diversity measure
        unique_values, counts = np.unique(decision_values, return_counts=True)
        probabilities = counts / len(decision_values)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(decision_values))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def ethical_alignment(agent, target_principles: Dict[str, float]) -> float:
        """Measure how well agent's decisions align with target ethical principles."""
        if not hasattr(agent, 'beliefs') or not agent.beliefs.beliefs:
            return 0.0
        
        alignment_scores = []
        for principle, target_value in target_principles.items():
            if principle in agent.beliefs.beliefs:
                agent_value = agent.beliefs.beliefs[principle]
                # Calculate alignment (1 - normalized absolute difference)
                alignment = 1.0 - abs(agent_value - target_value) / 2.0
                alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0

class LearningProgressMetrics:
    """Measures learning and adaptation progress."""
    
    @staticmethod
    def learning_rate(agent) -> float:
        """Measure how quickly an agent is learning/adapting."""
        if not hasattr(agent, 'decision_history') or len(agent.decision_history) < 5:
            return 0.0
        
        # Calculate change in decision patterns over time
        recent_decisions = agent.decision_history[-10:]
        older_decisions = agent.decision_history[-20:-10] if len(agent.decision_history) >= 20 else []
        
        if not older_decisions:
            return 0.0
        
        recent_avg = np.mean([d.get('value', 0.5) for d in recent_decisions])
        older_avg = np.mean([d.get('value', 0.5) for d in older_decisions])
        
        # Learning rate as absolute change
        return abs(recent_avg - older_avg)
    
    @staticmethod
    def adaptation_efficiency(agent) -> float:
        """Measure how efficiently an agent adapts to new scenarios."""
        if not hasattr(agent, 'decision_history') or len(agent.decision_history) < 3:
            return 0.0
        
        # Look for patterns in decision improvement
        decisions = agent.decision_history[-15:]
        if len(decisions) < 3:
            return 0.0
        
        # Calculate trend in decision confidence
        confidences = [d.get('confidence', 0.5) for d in decisions]
        if len(confidences) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(confidences))
        slope = np.polyfit(x, confidences, 1)[0]
        
        # Normalize slope to 0-1 range
        return min(1.0, max(0.0, slope + 0.5))

class MetricsCollector:
    """Main class for collecting and managing metrics."""
    
    def __init__(self):
        self.metric_history: List[MetricResult] = []
        self.ethical_metrics = EthicalConsistencyMetrics()
        self.social_metrics = SocialDynamicsMetrics()
        self.decision_metrics = DecisionQualityMetrics()
        self.learning_metrics = LearningProgressMetrics()
    
    def collect_agent_metrics(self, agent, agent_id: str = None) -> Dict[str, MetricResult]:
        """Collect all metrics for a single agent."""
        agent_id = agent_id or f"agent_{id(agent)}"
        metrics = {}
        
        # Ethical consistency metrics
        metrics['internal_consistency'] = MetricResult(
            name=f"{agent_id}_internal_consistency",
            value=self.ethical_metrics.internal_consistency(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Internal consistency of agent's beliefs"
        )
        
        metrics['temporal_stability'] = MetricResult(
            name=f"{agent_id}_temporal_stability",
            value=self.ethical_metrics.temporal_stability(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Stability of agent's beliefs over time"
        )
        
        metrics['cross_domain_consistency'] = MetricResult(
            name=f"{agent_id}_cross_domain_consistency",
            value=self.ethical_metrics.cross_domain_consistency(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Consistency across ethical domains"
        )
        
        # Decision quality metrics
        metrics['decision_confidence'] = MetricResult(
            name=f"{agent_id}_decision_confidence",
            value=self.decision_metrics.decision_confidence(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Confidence in recent decisions"
        )
        
        metrics['decision_diversity'] = MetricResult(
            name=f"{agent_id}_decision_diversity",
            value=self.decision_metrics.decision_diversity(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Diversity in decision-making patterns"
        )
        
        # Learning progress metrics
        metrics['learning_rate'] = MetricResult(
            name=f"{agent_id}_learning_rate",
            value=self.learning_metrics.learning_rate(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Rate of learning and adaptation"
        )
        
        metrics['adaptation_efficiency'] = MetricResult(
            name=f"{agent_id}_adaptation_efficiency",
            value=self.learning_metrics.adaptation_efficiency(agent),
            timestamp=datetime.now(),
            metadata={'agent_id': agent_id},
            description="Efficiency of adaptation to new scenarios"
        )
        
        # Store in history
        self.metric_history.extend(metrics.values())
        
        return metrics
    
    def collect_society_metrics(self, society) -> Dict[str, MetricResult]:
        """Collect all metrics for a society."""
        metrics = {}
        
        # Social dynamics metrics
        metrics['polarization'] = MetricResult(
            name="society_polarization",
            value=self.social_metrics.polarization_index(society),
            timestamp=datetime.now(),
            metadata={'n_agents': len(society.agents) if hasattr(society, 'agents') else 0},
            description="Level of polarization in society"
        )
        
        metrics['consensus'] = MetricResult(
            name="society_consensus",
            value=self.social_metrics.consensus_level(society),
            timestamp=datetime.now(),
            metadata={'n_agents': len(society.agents) if hasattr(society, 'agents') else 0},
            description="Level of consensus in society"
        )
        
        metrics['network_cohesion'] = MetricResult(
            name="society_network_cohesion",
            value=self.social_metrics.network_cohesion(society),
            timestamp=datetime.now(),
            metadata={'n_agents': len(society.agents) if hasattr(society, 'agents') else 0},
            description="Cohesion of social network"
        )
        
        # Influence distribution
        influence_dist = self.social_metrics.influence_distribution(society)
        for measure, value in influence_dist.items():
            metrics[f'influence_{measure}'] = MetricResult(
                name=f"society_influence_{measure}",
                value=value,
                timestamp=datetime.now(),
                metadata={'measure_type': measure},
                description=f"Influence distribution: {measure}"
            )
        
        # Store in history
        self.metric_history.extend(metrics.values())
        
        return metrics
    
    def collect_all_metrics(self, society) -> Dict[str, Any]:
        """Collect all metrics for society and its agents."""
        all_metrics = {}
        
        # Society metrics
        society_metrics = self.collect_society_metrics(society)
        all_metrics['society'] = society_metrics
        
        # Agent metrics
        all_metrics['agents'] = {}
        if hasattr(society, 'agents'):
            for i, agent in enumerate(society.agents):
                agent_id = f"agent_{i}"
                agent_metrics = self.collect_agent_metrics(agent, agent_id)
                all_metrics['agents'][agent_id] = agent_metrics
        
        return all_metrics
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export collected metrics to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = []
        for metric in self.metric_history:
            export_data.append({
                'name': metric.name,
                'value': metric.value,
                'timestamp': metric.timestamp.isoformat(),
                'metadata': metric.metadata,
                'description': metric.description
            })
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format.lower() == 'csv':
            if export_data:
                fieldnames = ['name', 'value', 'timestamp', 'description'] + \
                           list(export_data[0]['metadata'].keys())
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for row in export_data:
                        flat_row = {
                            'name': row['name'],
                            'value': row['value'],
                            'timestamp': row['timestamp'],
                            'description': row['description']
                        }
                        flat_row.update(row['metadata'])
                        writer.writerow(flat_row)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all collected metrics."""
        if not self.metric_history:
            return {"error": "No metrics collected yet"}
        
        # Group metrics by type
        metric_groups = {}
        for metric in self.metric_history:
            metric_type = metric.name.split('_')[1] if '_' in metric.name else 'other'
            if metric_type not in metric_groups:
                metric_groups[metric_type] = []
            metric_groups[metric_type].append(metric.value)
        
        # Calculate summary statistics
        summary = {
            'total_metrics': len(self.metric_history),
            'collection_period': {
                'start': min(m.timestamp for m in self.metric_history).isoformat(),
                'end': max(m.timestamp for m in self.metric_history).isoformat()
            },
            'metric_types': {}
        }
        
        for metric_type, values in metric_groups.items():
            summary['metric_types'][metric_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return summary
    
    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metric_history.clear()
