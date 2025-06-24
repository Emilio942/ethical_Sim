"""
Data export and automated reporting system for ethical agent simulations.

This module provides comprehensive data export capabilities and automated
report generation for simulation results, metrics, and analysis.
"""

import json
import csv
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    SQLITE = "sqlite"
    HTML = "html"
    MARKDOWN = "md"

class ReportType(Enum):
    """Types of reports that can be generated."""
    SIMULATION_SUMMARY = "simulation_summary"
    AGENT_ANALYSIS = "agent_analysis"
    SOCIETY_DYNAMICS = "society_dynamics"
    METRICS_REPORT = "metrics_report"
    VALIDATION_REPORT = "validation_report"
    COMPARATIVE_ANALYSIS = "comparative_analysis"

@dataclass
class ExportConfig:
    """Configuration for data export."""
    format: ExportFormat
    filepath: str
    include_metadata: bool = True
    include_raw_data: bool = True
    include_summaries: bool = True
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"
    compression: Optional[str] = None  # "gzip", "bz2", etc.

class DataExporter:
    """Handles export of simulation data in various formats."""
    
    @staticmethod
    def export_society_data(society, config: ExportConfig) -> str:
        """Export complete society data."""
        filepath = Path(config.filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare society data
        society_data = DataExporter._extract_society_data(society)
        
        if config.format == ExportFormat.JSON:
            return DataExporter._export_json(society_data, filepath, config)
        elif config.format == ExportFormat.CSV:
            return DataExporter._export_csv(society_data, filepath, config)
        elif config.format == ExportFormat.PICKLE:
            return DataExporter._export_pickle(society_data, filepath, config)
        elif config.format == ExportFormat.SQLITE:
            return DataExporter._export_sqlite(society_data, filepath, config)
        else:
            raise ValueError(f"Unsupported export format: {config.format}")
    
    @staticmethod
    def export_agent_data(agent, agent_id: str, config: ExportConfig) -> str:
        """Export individual agent data."""
        filepath = Path(config.filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare agent data
        agent_data = DataExporter._extract_agent_data(agent, agent_id)
        
        if config.format == ExportFormat.JSON:
            return DataExporter._export_json(agent_data, filepath, config)
        elif config.format == ExportFormat.CSV:
            return DataExporter._export_csv_agent(agent_data, filepath, config)
        else:
            return DataExporter.export_society_data({'agents': [agent_data]}, config)
    
    @staticmethod
    def export_metrics_data(metrics_data: Dict[str, Any], config: ExportConfig) -> str:
        """Export metrics data."""
        filepath = Path(config.filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if config.format == ExportFormat.JSON:
            return DataExporter._export_json(metrics_data, filepath, config)
        elif config.format == ExportFormat.CSV:
            return DataExporter._export_metrics_csv(metrics_data, filepath, config)
        else:
            return DataExporter.export_society_data(metrics_data, config)
    
    @staticmethod
    def _extract_society_data(society) -> Dict[str, Any]:
        """Extract comprehensive data from society."""
        data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'society_type': type(society).__name__,
                'n_agents': len(society.agents) if hasattr(society, 'agents') else 0
            },
            'agents': [],
            'social_network': {},
            'group_dynamics': {},
            'simulation_state': {}
        }
        
        # Extract agent data
        if hasattr(society, 'agents'):
            for i, agent in enumerate(society.agents):
                agent_data = DataExporter._extract_agent_data(agent, f"agent_{i}")
                data['agents'].append(agent_data)
        
        # Extract social network data
        if hasattr(society, 'social_network'):
            G = society.social_network
            data['social_network'] = {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'edges': list(G.edges(data=True)),
                'nodes': {str(node): G.nodes[node] for node in G.nodes()},
                'is_directed': G.is_directed()
            }
        
        # Extract group dynamics
        if hasattr(society, 'groups'):
            data['group_dynamics']['groups'] = society.groups
        if hasattr(society, 'group_cohesion'):
            data['group_dynamics']['cohesion'] = society.group_cohesion
        if hasattr(society, 'opinion_leaders'):
            data['group_dynamics']['opinion_leaders'] = society.opinion_leaders
        
        # Extract simulation state
        if hasattr(society, 'simulation_step'):
            data['simulation_state']['step'] = society.simulation_step
        if hasattr(society, 'last_scenario'):
            data['simulation_state']['last_scenario'] = society.last_scenario
        
        return data
    
    @staticmethod
    def _extract_agent_data(agent, agent_id: str) -> Dict[str, Any]:
        """Extract comprehensive data from an agent."""
        data = {
            'agent_id': agent_id,
            'agent_type': type(agent).__name__,
            'beliefs': {},
            'decision_history': [],
            'personality': {},
            'learning_state': {},
            'social_connections': {}
        }
        
        # Extract beliefs
        if hasattr(agent, 'beliefs') and agent.beliefs:
            if hasattr(agent.beliefs, 'beliefs'):
                data['beliefs'] = dict(agent.beliefs.beliefs)
            if hasattr(agent.beliefs, 'uncertainty'):
                data['beliefs']['_uncertainty'] = agent.beliefs.uncertainty
            if hasattr(agent.beliefs, 'confidence'):
                data['beliefs']['_confidence'] = agent.beliefs.confidence
        
        # Extract decision history
        if hasattr(agent, 'decision_history'):
            data['decision_history'] = [
                {
                    'timestamp': d.get('timestamp', ''),
                    'scenario': d.get('scenario', ''),
                    'value': d.get('value', 0.5),
                    'confidence': d.get('confidence', 0.5),
                    'reasoning': d.get('reasoning', ''),
                    'learning_mode': d.get('learning_mode', '')
                } for d in agent.decision_history
            ]
        
        # Extract personality traits
        if hasattr(agent, 'personality'):
            data['personality'] = dict(agent.personality)
        
        # Extract learning state
        if hasattr(agent, 'q_table'):
            data['learning_state']['q_table'] = dict(agent.q_table)
        if hasattr(agent, 'learning_rate'):
            data['learning_state']['learning_rate'] = agent.learning_rate
        if hasattr(agent, 'confidence_threshold'):
            data['learning_state']['confidence_threshold'] = agent.confidence_threshold
        
        # Extract social connections
        if hasattr(agent, 'trust_network'):
            data['social_connections']['trust_network'] = dict(agent.trust_network)
        if hasattr(agent, 'reputation_scores'):
            data['social_connections']['reputation_scores'] = dict(agent.reputation_scores)
        
        return data
    
    @staticmethod
    def _export_json(data: Dict[str, Any], filepath: Path, config: ExportConfig) -> str:
        """Export data as JSON."""
        def json_serializer(obj):
            """Custom JSON serializer for numpy types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=json_serializer, ensure_ascii=False)
        
        return str(filepath)
    
    @staticmethod
    def _export_csv(data: Dict[str, Any], filepath: Path, config: ExportConfig) -> str:
        """Export data as CSV (flattened structure)."""
        # Create multiple CSV files for complex data
        base_path = filepath.with_suffix('')
        exported_files = []
        
        # Export agents data
        if 'agents' in data and data['agents']:
            agents_file = f"{base_path}_agents.csv"
            DataExporter._export_agents_csv(data['agents'], agents_file)
            exported_files.append(agents_file)
        
        # Export social network data
        if 'social_network' in data and data['social_network'].get('edges'):
            network_file = f"{base_path}_network.csv"
            DataExporter._export_network_csv(data['social_network'], network_file)
            exported_files.append(network_file)
        
        # Export metadata
        metadata_file = f"{base_path}_metadata.csv"
        DataExporter._export_metadata_csv(data.get('metadata', {}), metadata_file)
        exported_files.append(metadata_file)
        
        return f"Exported {len(exported_files)} CSV files: {', '.join(exported_files)}"
    
    @staticmethod
    def _export_agents_csv(agents_data: List[Dict], filepath: str) -> None:
        """Export agents data to CSV."""
        if not agents_data:
            return
        
        # Flatten agent data for CSV
        rows = []
        for agent in agents_data:
            row = {
                'agent_id': agent.get('agent_id', ''),
                'agent_type': agent.get('agent_type', ''),
                'n_beliefs': len(agent.get('beliefs', {})),
                'n_decisions': len(agent.get('decision_history', [])),
                'avg_decision_value': np.mean([d.get('value', 0.5) 
                                             for d in agent.get('decision_history', [])]) 
                                     if agent.get('decision_history') else 0.0,
                'avg_confidence': np.mean([d.get('confidence', 0.5) 
                                         for d in agent.get('decision_history', [])]) 
                                if agent.get('decision_history') else 0.0
            }
            
            # Add belief values
            beliefs = agent.get('beliefs', {})
            for belief, value in beliefs.items():
                if not belief.startswith('_'):  # Skip metadata
                    row[f'belief_{belief}'] = value
            
            # Add personality traits
            personality = agent.get('personality', {})
            for trait, value in personality.items():
                row[f'personality_{trait}'] = value
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = set()
            for row in rows:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    @staticmethod
    def _export_network_csv(network_data: Dict, filepath: str) -> None:
        """Export social network data to CSV."""
        edges = network_data.get('edges', [])
        if not edges:
            return
        
        rows = []
        for edge in edges:
            if len(edge) >= 2:
                row = {
                    'source': edge[0],
                    'target': edge[1],
                    'weight': edge[2].get('weight', 1.0) if len(edge) > 2 else 1.0
                }
                # Add other edge attributes
                if len(edge) > 2:
                    for key, value in edge[2].items():
                        if key != 'weight':
                            row[key] = value
                rows.append(row)
        
        if rows:
            fieldnames = set()
            for row in rows:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    @staticmethod
    def _export_metadata_csv(metadata: Dict, filepath: str) -> None:
        """Export metadata to CSV."""
        rows = [{'key': k, 'value': v} for k, v in metadata.items()]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'value'])
            writer.writeheader()
            writer.writerows(rows)
    
    @staticmethod
    def _export_csv_agent(agent_data: Dict, filepath: Path, config: ExportConfig) -> str:
        """Export single agent data as CSV."""
        decision_history = agent_data.get('decision_history', [])
        if not decision_history:
            return "No decision history to export"
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'scenario', 'value', 'confidence', 'reasoning', 'learning_mode']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(decision_history)
        
        return str(filepath)
    
    @staticmethod
    def _export_metrics_csv(metrics_data: Dict, filepath: Path, config: ExportConfig) -> str:
        """Export metrics data as CSV."""
        rows = []
        
        def flatten_metrics(data, prefix=''):
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value:
                    # This is a metric result
                    row = {
                        'metric_name': f"{prefix}{key}",
                        'value': value.get('value', 0),
                        'timestamp': value.get('timestamp', ''),
                        'description': value.get('description', '')
                    }
                    # Add metadata fields
                    metadata = value.get('metadata', {})
                    for meta_key, meta_value in metadata.items():
                        row[f'meta_{meta_key}'] = meta_value
                    rows.append(row)
                elif isinstance(value, dict):
                    flatten_metrics(value, f"{prefix}{key}_")
        
        flatten_metrics(metrics_data)
        
        if rows:
            fieldnames = set()
            for row in rows:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return str(filepath)
    
    @staticmethod
    def _export_pickle(data: Dict[str, Any], filepath: Path, config: ExportConfig) -> str:
        """Export data as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return str(filepath)
    
    @staticmethod
    def _export_sqlite(data: Dict[str, Any], filepath: Path, config: ExportConfig) -> str:
        """Export data to SQLite database."""
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        
        try:
            # Create tables
            DataExporter._create_sqlite_tables(cursor)
            
            # Insert metadata
            metadata = data.get('metadata', {})
            for key, value in metadata.items():
                cursor.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    (key, str(value))
                )
            
            # Insert agents
            agents = data.get('agents', [])
            for agent in agents:
                agent_id = agent.get('agent_id', '')
                cursor.execute(
                    "INSERT INTO agents (agent_id, agent_type, beliefs_json, personality_json) VALUES (?, ?, ?, ?)",
                    (
                        agent_id,
                        agent.get('agent_type', ''),
                        json.dumps(agent.get('beliefs', {})),
                        json.dumps(agent.get('personality', {}))
                    )
                )
                
                # Insert decision history
                for decision in agent.get('decision_history', []):
                    cursor.execute(
                        "INSERT INTO decisions (agent_id, timestamp, scenario, value, confidence, reasoning) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            agent_id,
                            decision.get('timestamp', ''),
                            decision.get('scenario', ''),
                            decision.get('value', 0.5),
                            decision.get('confidence', 0.5),
                            decision.get('reasoning', '')
                        )
                    )
            
            # Insert social network
            network = data.get('social_network', {})
            for edge in network.get('edges', []):
                if len(edge) >= 2:
                    cursor.execute(
                        "INSERT INTO social_network (source, target, weight, attributes_json) VALUES (?, ?, ?, ?)",
                        (
                            str(edge[0]),
                            str(edge[1]),
                            edge[2].get('weight', 1.0) if len(edge) > 2 else 1.0,
                            json.dumps(edge[2]) if len(edge) > 2 else '{}'
                        )
                    )
            
            conn.commit()
            return str(filepath)
        
        finally:
            conn.close()
    
    @staticmethod
    def _create_sqlite_tables(cursor) -> None:
        """Create SQLite tables for simulation data."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT,
                beliefs_json TEXT,
                personality_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                timestamp TEXT,
                scenario TEXT,
                value REAL,
                confidence REAL,
                reasoning TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents (agent_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS social_network (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                weight REAL,
                attributes_json TEXT
            )
        ''')

class ReportGenerator:
    """Generates automated reports from simulation data."""
    
    @staticmethod
    def generate_simulation_summary(society, metrics_data=None, validation_data=None) -> str:
        """Generate a comprehensive simulation summary report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Ethical Agent Simulation Report
Generated: {timestamp}

## Executive Summary
"""
        
        # Society overview
        n_agents = len(society.agents) if hasattr(society, 'agents') else 0
        report += f"- **Total Agents**: {n_agents}\n"
        
        if hasattr(society, 'social_network'):
            G = society.social_network
            report += f"- **Social Network**: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
            if G.number_of_nodes() > 1:
                density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
                report += f"- **Network Density**: {density:.3f}\n"
        
        # Agent statistics
        if hasattr(society, 'agents') and society.agents:
            total_decisions = sum(len(agent.decision_history) if hasattr(agent, 'decision_history') else 0 
                                for agent in society.agents)
            report += f"- **Total Decisions Made**: {total_decisions}\n"
            
            if total_decisions > 0:
                avg_decisions = total_decisions / len(society.agents)
                report += f"- **Average Decisions per Agent**: {avg_decisions:.1f}\n"
        
        report += "\n## Agent Analysis\n"
        
        # Individual agent summaries
        if hasattr(society, 'agents'):
            for i, agent in enumerate(society.agents):
                report += f"\n### Agent {i}\n"
                
                # Beliefs summary
                if hasattr(agent, 'beliefs') and agent.beliefs.beliefs:
                    belief_count = len(agent.beliefs.beliefs)
                    avg_belief = np.mean(list(agent.beliefs.beliefs.values()))
                    report += f"- **Beliefs**: {belief_count} beliefs, average strength: {avg_belief:.2f}\n"
                
                # Decision history
                if hasattr(agent, 'decision_history'):
                    decision_count = len(agent.decision_history)
                    if decision_count > 0:
                        avg_value = np.mean([d.get('value', 0.5) for d in agent.decision_history])
                        avg_confidence = np.mean([d.get('confidence', 0.5) for d in agent.decision_history])
                        report += f"- **Decisions**: {decision_count} decisions, avg value: {avg_value:.2f}, avg confidence: {avg_confidence:.2f}\n"
                
                # Personality
                if hasattr(agent, 'personality'):
                    personality_traits = len(agent.personality)
                    report += f"- **Personality**: {personality_traits} traits\n"
        
        # Metrics summary
        if metrics_data:
            report += "\n## Metrics Summary\n"
            
            if 'society' in metrics_data:
                society_metrics = metrics_data['society']
                for metric_name, metric_result in society_metrics.items():
                    if hasattr(metric_result, 'value'):
                        report += f"- **{metric_name}**: {metric_result.value:.3f}\n"
        
        # Validation summary
        if validation_data:
            report += "\n## Validation Results\n"
            
            total_checks = 0
            passed_checks = 0
            
            for category, results in validation_data.items():
                if isinstance(results, list):
                    total_checks += len(results)
                    passed_checks += sum(1 for r in results if r.passed)
                elif isinstance(results, dict):
                    for agent_results in results.values():
                        if isinstance(agent_results, list):
                            total_checks += len(agent_results)
                            passed_checks += sum(1 for r in agent_results if r.passed)
            
            if total_checks > 0:
                pass_rate = passed_checks / total_checks
                report += f"- **Overall Pass Rate**: {pass_rate:.1%} ({passed_checks}/{total_checks})\n"
        
        return report
    
    @staticmethod
    def generate_agent_analysis(agent, agent_id: str, metrics_data=None) -> str:
        """Generate detailed analysis for a single agent."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Agent Analysis Report: {agent_id}
Generated: {timestamp}

## Agent Overview
- **Agent Type**: {type(agent).__name__}
- **Agent ID**: {agent_id}

"""
        
        # Beliefs analysis
        if hasattr(agent, 'beliefs') and agent.beliefs.beliefs:
            report += "## Belief System\n"
            beliefs = agent.beliefs.beliefs
            
            report += f"- **Total Beliefs**: {len(beliefs)}\n"
            belief_values = list(beliefs.values())
            report += f"- **Belief Strength Range**: {min(belief_values):.2f} to {max(belief_values):.2f}\n"
            report += f"- **Average Belief Strength**: {np.mean(belief_values):.2f}\n"
            report += f"- **Belief Variance**: {np.var(belief_values):.3f}\n"
            
            # Top beliefs
            sorted_beliefs = sorted(beliefs.items(), key=lambda x: abs(x[1]), reverse=True)
            report += "\n### Strongest Beliefs\n"
            for belief, strength in sorted_beliefs[:5]:
                report += f"- **{belief}**: {strength:.2f}\n"
        
        # Decision history analysis
        if hasattr(agent, 'decision_history') and agent.decision_history:
            report += "\n## Decision History\n"
            decisions = agent.decision_history
            
            report += f"- **Total Decisions**: {len(decisions)}\n"
            
            decision_values = [d.get('value', 0.5) for d in decisions]
            confidences = [d.get('confidence', 0.5) for d in decisions]
            
            report += f"- **Average Decision Value**: {np.mean(decision_values):.2f}\n"
            report += f"- **Decision Variance**: {np.var(decision_values):.3f}\n"
            report += f"- **Average Confidence**: {np.mean(confidences):.2f}\n"
            
            # Recent decision trend
            if len(decisions) >= 10:
                recent_values = decision_values[-10:]
                older_values = decision_values[-20:-10] if len(decisions) >= 20 else decision_values[:-10]
                
                if older_values:
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    trend = recent_avg - older_avg
                    report += f"- **Recent Trend**: {trend:+.3f} (change in last 10 decisions)\n"
        
        # Personality analysis
        if hasattr(agent, 'personality'):
            report += "\n## Personality Profile\n"
            for trait, value in agent.personality.items():
                report += f"- **{trait}**: {value:.2f}\n"
        
        # Learning analysis
        learning_info = []
        if hasattr(agent, 'q_table'):
            learning_info.append(f"Q-Learning table size: {len(agent.q_table)}")
        if hasattr(agent, 'learning_rate'):
            learning_info.append(f"Learning rate: {agent.learning_rate:.3f}")
        if hasattr(agent, 'confidence_threshold'):
            learning_info.append(f"Confidence threshold: {agent.confidence_threshold:.3f}")
        
        if learning_info:
            report += "\n## Learning State\n"
            for info in learning_info:
                report += f"- {info}\n"
        
        # Metrics analysis
        if metrics_data and agent_id in metrics_data:
            report += "\n## Performance Metrics\n"
            agent_metrics = metrics_data[agent_id]
            for metric_name, metric_result in agent_metrics.items():
                if hasattr(metric_result, 'value'):
                    report += f"- **{metric_name}**: {metric_result.value:.3f}\n"
        
        return report
    
    @staticmethod
    def generate_html_report(report_content: str, title: str = "Simulation Report") -> str:
        """Convert markdown report to HTML."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #5a6c7d;
            margin-top: 20px;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        li {{
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        strong {{
            color: #2c3e50;
        }}
        .metric {{
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        {ReportGenerator._markdown_to_html(report_content)}
    </div>
</body>
</html>
"""
        return html_template
    
    @staticmethod
    def _markdown_to_html(markdown_text: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown_text
        
        # Headers
        html = html.replace('# ', '<h1>')
        html = html.replace('\n## ', '</h1>\n<h2>')
        html = html.replace('\n### ', '</h2>\n<h3>')
        
        # Close last header
        if '<h1>' in html and '</h1>' not in html:
            html += '</h1>'
        if '<h2>' in html and '</h2>' not in html:
            html += '</h2>'
        if '<h3>' in html and '</h3>' not in html:
            html += '</h3>'
        
        # Bold text
        html = html.replace('**', '<strong>', 1)
        html = html.replace('**', '</strong>', 1)
        
        # Line breaks
        html = html.replace('\n\n', '</p><p>')
        html = '<p>' + html + '</p>'
        
        # Lists
        html = html.replace('- ', '<li>')
        html = html.replace('<li>', '<ul><li>', 1)
        html = html.replace('</p><p><li>', '</ul></p><p><ul><li>')
        html += '</ul>'
        
        return html

class AutomatedReporter:
    """Orchestrates automated report generation."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, society, metrics_data=None, validation_data=None) -> Dict[str, str]:
        """Generate all types of reports for a simulation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files = {}
        
        # Simulation summary
        summary_report = ReportGenerator.generate_simulation_summary(society, metrics_data, validation_data)
        
        # Save as markdown
        md_file = self.output_dir / f"simulation_summary_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        generated_files['summary_md'] = str(md_file)
        
        # Save as HTML
        html_report = ReportGenerator.generate_html_report(summary_report, "Simulation Summary")
        html_file = self.output_dir / f"simulation_summary_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        generated_files['summary_html'] = str(html_file)
        
        # Individual agent reports
        if hasattr(society, 'agents'):
            agents_dir = self.output_dir / f"agents_{timestamp}"
            agents_dir.mkdir(exist_ok=True)
            
            for i, agent in enumerate(society.agents):
                agent_id = f"agent_{i}"
                agent_metrics = metrics_data.get('agents', {}).get(agent_id) if metrics_data else None
                
                agent_report = ReportGenerator.generate_agent_analysis(agent, agent_id, agent_metrics)
                
                agent_file = agents_dir / f"{agent_id}_analysis.md"
                with open(agent_file, 'w', encoding='utf-8') as f:
                    f.write(agent_report)
        
        # Export raw data
        export_config = ExportConfig(
            format=ExportFormat.JSON,
            filepath=str(self.output_dir / f"simulation_data_{timestamp}.json")
        )
        data_file = DataExporter.export_society_data(society, export_config)
        generated_files['data_json'] = data_file
        
        # Export metrics if available
        if metrics_data:
            metrics_config = ExportConfig(
                format=ExportFormat.JSON,
                filepath=str(self.output_dir / f"metrics_data_{timestamp}.json")
            )
            metrics_file = DataExporter.export_metrics_data(metrics_data, metrics_config)
            generated_files['metrics_json'] = metrics_file
        
        return generated_files
    
    def schedule_periodic_reports(self, society, interval_hours: int = 24) -> None:
        """Schedule periodic report generation (placeholder for future implementation)."""
        # This would integrate with a task scheduler in a production environment
        pass
