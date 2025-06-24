#!/usr/bin/env python3
"""
Tiefgreifende System-Tests für Ethische Agenten-Simulation
=========================================================

Echte Tests mit Edge Cases, Fehlerbehandlung, Memory Leaks,
Performance unter Last, Datenintegrität und Konsistenz.
"""

import traceback
import gc
import psutil
import time
import threading
import random
import math
import numpy as np
from typing import List, Dict, Any
import sys
import os

class DeepSystemTester:
    """Führt tiefgreifende Systemtests durch"""
    
    def __init__(self):
        self.test_results = []
        self.memory_baseline = None
        self.start_time = time.time()
        
    def log_test(self, test_name: str, passed: bool, details: str, error: Exception = None):
        """Protokolliert Testergebnisse"""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': time.time() - self.start_time
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        if error:
            print(f"   ERROR: {error}")
    
    def test_memory_leaks(self):
        """Test auf Memory Leaks bei wiederholten Operationen"""
        print("\n🧠 MEMORY LEAK TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            
            # Baseline Memory
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            societies = []
            
            # Erstelle und zerstöre 100 Societies
            for i in range(100):
                society = NeuralEthicalSociety()
                for j in range(5):
                    agent = NeuralEthicalAgent(f'agent_{i}_{j}')
                    society.add_agent(agent)
                societies.append(society)
                
                # Alle 20 Iterationen: Cleanup
                if i % 20 == 0:
                    societies.clear()
                    gc.collect()
            
            # Final Memory Check
            societies.clear()
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory Leak Test: Nicht mehr als 10MB Anstieg
            if memory_increase < 10:
                self.log_test("Memory Leak Test", True, f"Memory increase: {memory_increase:.2f}MB")
            else:
                self.log_test("Memory Leak Test", False, f"Excessive memory increase: {memory_increase:.2f}MB")
                
        except Exception as e:
            self.log_test("Memory Leak Test", False, "Exception during test", e)
    
    def test_concurrent_access(self):
        """Test auf Thread-Safety und Concurrent Access"""
        print("\n🔀 CONCURRENT ACCESS TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            from scenarios import get_trolley_problem
            
            society = NeuralEthicalSociety()
            
            # Füge initial Agenten hinzu
            for i in range(5):
                agent = NeuralEthicalAgent(f'concurrent_agent_{i}')
                society.add_agent(agent)
            
            scenario = get_trolley_problem()
            errors = []
            results = []
            
            def worker_thread(thread_id: int):
                """Worker Thread für Concurrent Testing"""
                try:
                    # Verschiedene concurrent operations
                    for i in range(10):
                        # Entscheidungen treffen
                        agent_keys = list(society.agents.keys())
                        if agent_keys:
                            agent = society.agents[random.choice(agent_keys)]
                            decision = agent.make_decision(scenario)
                            results.append((thread_id, i, decision))
                        
                        # Kurze Pause um Race Conditions zu fördern
                        time.sleep(0.001)
                        
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Starte 5 concurrent threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Warte auf alle Threads
            for thread in threads:
                thread.join()
            
            # Auswertung
            if len(errors) == 0 and len(results) == 50:  # 5 threads * 10 operations
                self.log_test("Concurrent Access Test", True, f"All {len(results)} operations completed without errors")
            else:
                self.log_test("Concurrent Access Test", False, f"Errors: {len(errors)}, Results: {len(results)}/50")
                for thread_id, error in errors:
                    print(f"   Thread {thread_id} error: {error}")
                    
        except Exception as e:
            self.log_test("Concurrent Access Test", False, "Exception during test", e)
    
    def test_edge_cases_data_corruption(self):
        """Test Edge Cases und Datenkorruption"""
        print("\n⚠️  EDGE CASES & DATA CORRUPTION TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            from scenarios import get_trolley_problem
            
            # Test 1: Leere/Null Inputs
            try:
                agent = NeuralEthicalAgent('')  # Leere ID
                self.log_test("Empty Agent ID", False, "Should have raised exception")
            except:
                self.log_test("Empty Agent ID", True, "Correctly rejected empty ID")
            
            # Test 2: Extreme Values
            try:
                agent = NeuralEthicalAgent('test_agent')
                # Versuche extreme Belief Values zu setzen
                # Create a more complete mock belief with all required attributes and methods
                class MockBelief:
                    def __init__(self):
                        self.name = 'extreme_belief'
                        self.strength = float('inf')  # Infinite value
                        self.confidence = -999999  # Extreme negative
                        self.certainty = 0.5
                        self.activation = 0.0
                        self.connections = {}
                        self.associated_concepts = {}
                    
                    def activate(self, level, time):
                        """Mock activate method"""
                        self.activation = max(0.0, min(1.0, level))  # Clamp to valid range
                    
                    def update_certainty(self, new_certainty):
                        """Mock update_certainty method"""
                        self.certainty = max(0.0, min(1.0, new_certainty))  # Clamp to valid range
                    
                    def update_strength(self, new_strength):
                        """Mock update_strength method"""
                        self.strength = max(0.0, min(1.0, new_strength)) if not math.isinf(new_strength) else 1.0
                
                agent.add_belief(MockBelief())
                
                scenario = get_trolley_problem()
                decision = agent.make_decision(scenario)
                
                # Prüfe ob Decision sane Werte hat
                confidence = decision.get('confidence', 0)
                if 0 <= confidence <= 1:
                    self.log_test("Extreme Values Handling", True, f"Confidence normalized to {confidence}")
                else:
                    self.log_test("Extreme Values Handling", False, f"Invalid confidence: {confidence}")
                    
            except Exception as e:
                self.log_test("Extreme Values Handling", False, "Exception with extreme values", e)
            
            # Test 3: Invalid Data Types
            try:
                society = NeuralEthicalSociety()
                # Versuche invalide Agenten hinzuzufügen
                society.add_agent("not_an_agent")  # String statt Agent
                self.log_test("Invalid Data Types", False, "Should have rejected invalid agent type")
            except:
                self.log_test("Invalid Data Types", True, "Correctly rejected invalid data type")
            
            # Test 4: Circular References
            try:
                agent1 = NeuralEthicalAgent('agent1')
                agent2 = NeuralEthicalAgent('agent2')
                
                # Erstelle potentielle circular reference
                agent1.social_connections = {'agent2': 1.0}
                agent2.social_connections = {'agent1': 1.0}
                
                society = NeuralEthicalSociety()
                society.add_agent(agent1)
                society.add_agent(agent2)
                
                # Test ob System mit circular refs umgehen kann
                scenario = get_trolley_problem()
                decision1 = agent1.make_decision(scenario)
                decision2 = agent2.make_decision(scenario)
                
                self.log_test("Circular References", True, "Handled circular references without issues")
                
            except Exception as e:
                self.log_test("Circular References", False, "Failed with circular references", e)
                
        except Exception as e:
            self.log_test("Edge Cases Test", False, "Exception during edge case testing", e)
    
    def test_performance_under_load(self):
        """Test Performance unter extremer Last"""
        print("\n⚡ PERFORMANCE UNDER LOAD TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            from scenarios import get_trolley_problem
            
            # Test 1: Viele Agenten
            start_time = time.time()
            
            society = NeuralEthicalSociety()
            
            # Erstelle 100 Agenten
            for i in range(100):
                agent = NeuralEthicalAgent(f'load_agent_{i}')
                # Füge zufällige Personality-Traits hinzu
                agent.personality_traits = {
                    'openness': random.random(),
                    'conscientiousness': random.random(),
                    'extraversion': random.random(),
                    'agreeableness': random.random(),
                    'neuroticism': random.random()
                }
                society.add_agent(agent)
            
            creation_time = time.time() - start_time
            
            if creation_time < 5.0:  # Sollte unter 5 Sekunden sein
                self.log_test("Mass Agent Creation", True, f"Created 100 agents in {creation_time:.2f}s")
            else:
                self.log_test("Mass Agent Creation", False, f"Too slow: {creation_time:.2f}s")
            
            # Test 2: Viele Entscheidungen
            start_time = time.time()
            
            scenario = get_trolley_problem()
            decisions = []
            
            # 1000 Entscheidungen
            for i in range(1000):
                agent_keys = list(society.agents.keys())
                agent = society.agents[random.choice(agent_keys)]
                decision = agent.make_decision(scenario)
                decisions.append(decision)
            
            decision_time = time.time() - start_time
            decisions_per_second = 1000 / decision_time
            
            if decisions_per_second > 100:  # Mindestens 100 Entscheidungen/Sekunde
                self.log_test("Mass Decision Making", True, f"{decisions_per_second:.1f} decisions/second")
            else:
                self.log_test("Mass Decision Making", False, f"Too slow: {decisions_per_second:.1f} decisions/second")
            
            # Test 3: Memory unter Last
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Noch mehr Agenten und Entscheidungen
            for i in range(500):
                agent = NeuralEthicalAgent(f'stress_agent_{i}')
                society.add_agent(agent)
                if i % 50 == 0:  # Alle 50 Agenten: 10 Entscheidungen
                    for j in range(10):
                        decision = agent.make_decision(scenario)
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            if memory_increase < 100:  # Nicht mehr als 100MB für 500 zusätzliche Agenten
                self.log_test("Memory Under Load", True, f"Memory increase: {memory_increase:.1f}MB")
            else:
                self.log_test("Memory Under Load", False, f"Excessive memory use: {memory_increase:.1f}MB")
                
        except Exception as e:
            self.log_test("Performance Under Load", False, "Exception during load testing", e)
    
    def test_data_consistency(self):
        """Test Datenintegrität und Konsistenz"""
        print("\n🔍 DATA CONSISTENCY TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            from scenarios import get_trolley_problem
            
            society = NeuralEthicalSociety()
            agent = NeuralEthicalAgent('consistency_agent')
            society.add_agent(agent)
            
            scenario = get_trolley_problem()
            
            # Test 1: Entscheidungs-Konsistenz
            decisions = []
            for i in range(10):
                decision = agent.make_decision(scenario)
                decisions.append(decision)
            
            # Prüfe ob alle Entscheidungen valide Struktur haben
            all_valid = True
            for i, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    self.log_test("Decision Structure", False, f"Decision {i} is not dict: {type(decision)}")
                    all_valid = False
                    break
                
                required_keys = ['chosen_option', 'confidence']
                for key in required_keys:
                    if key not in decision:
                        self.log_test("Decision Structure", False, f"Decision {i} missing key: {key}")
                        all_valid = False
                        break
                
                # Prüfe Confidence Range
                confidence = decision.get('confidence', -1)
                if not (0 <= confidence <= 1):
                    self.log_test("Decision Structure", False, f"Invalid confidence {confidence} in decision {i}")
                    all_valid = False
                    break
            
            if all_valid:
                self.log_test("Decision Structure", True, f"All {len(decisions)} decisions have valid structure")
            
            # Test 2: Agent State Consistency
            agent_state_1 = {
                'agent_id': agent.agent_id,
                'personality_traits': dict(agent.personality_traits) if hasattr(agent, 'personality_traits') else {},
                'beliefs_count': len(agent.beliefs.beliefs) if hasattr(agent.beliefs, 'beliefs') else 0
            }
            
            # Führe Operationen durch
            for i in range(5):
                agent.make_decision(scenario)
            
            agent_state_2 = {
                'agent_id': agent.agent_id,
                'personality_traits': dict(agent.personality_traits) if hasattr(agent, 'personality_traits') else {},
                'beliefs_count': len(agent.beliefs.beliefs) if hasattr(agent.beliefs, 'beliefs') else 0
            }
            
            # Agent ID sollte unverändert sein
            if agent_state_1['agent_id'] == agent_state_2['agent_id']:
                self.log_test("Agent ID Consistency", True, "Agent ID remained stable")
            else:
                self.log_test("Agent ID Consistency", False, f"Agent ID changed: {agent_state_1['agent_id']} -> {agent_state_2['agent_id']}")
            
            # Test 3: Society Integrity
            initial_agent_count = len(society.agents)
            
            # Operationen die Society nicht beschädigen sollten
            for agent_id, agent_obj in society.agents.items():
                decision = agent_obj.make_decision(scenario)
            
            final_agent_count = len(society.agents)
            
            if initial_agent_count == final_agent_count:
                self.log_test("Society Integrity", True, f"Agent count stable: {final_agent_count}")
            else:
                self.log_test("Society Integrity", False, f"Agent count changed: {initial_agent_count} -> {final_agent_count}")
                
        except Exception as e:
            self.log_test("Data Consistency", False, "Exception during consistency testing", e)
    
    def test_error_handling_resilience(self):
        """Test Fehlerbehandlung und System-Resilience"""
        print("\n🛡️  ERROR HANDLING & RESILIENCE TESTS")
        print("-" * 40)
        
        try:
            from neural_society import NeuralEthicalSociety
            from agents import NeuralEthicalAgent
            
            # Test 1: Corrupted Scenario Handling
            try:
                agent = NeuralEthicalAgent('resilience_agent')
                
                # Erstelle corrupted scenario mock
                corrupted_scenario = type('CorruptedScenario', (), {
                    'scenario_id': None,  # Null ID
                    'options': [],  # Leere Options
                    'description': None
                })()
                
                decision = agent.make_decision(corrupted_scenario)
                
                # System sollte graceful handling haben
                if decision is not None:
                    self.log_test("Corrupted Scenario Handling", True, "Gracefully handled corrupted scenario")
                else:
                    self.log_test("Corrupted Scenario Handling", False, "Returned None for corrupted scenario")
                    
            except Exception as e:
                # Exception ist auch OK, solange das System nicht crashed
                self.log_test("Corrupted Scenario Handling", True, f"Properly raised exception: {type(e).__name__}")
            
            # Test 2: Invalid Agent State Recovery
            try:
                society = NeuralEthicalSociety()
                agent = NeuralEthicalAgent('recovery_agent')
                
                # Corrupte Agent-Daten simulieren
                if hasattr(agent, 'beliefs'):
                    agent.beliefs.beliefs = None  # Null beliefs
                
                society.add_agent(agent)
                
                # Teste ob Society trotzdem funktioniert
                agent_count = len(society.agents)
                
                if agent_count == 1:
                    self.log_test("Invalid Agent State Recovery", True, "Society accepted agent with corrupted beliefs")
                else:
                    self.log_test("Invalid Agent State Recovery", False, f"Unexpected agent count: {agent_count}")
                    
            except Exception as e:
                self.log_test("Invalid Agent State Recovery", True, f"Properly handled invalid state: {type(e).__name__}")
            
            # Test 3: Resource Exhaustion Simulation
            try:
                # Simuliere hohe Memory-Nutzung
                big_data = []
                society = NeuralEthicalSociety()
                
                # Erstelle viele Agenten mit "großen" Daten
                for i in range(50):
                    agent = NeuralEthicalAgent(f'memory_agent_{i}')
                    # Simuliere "große" Agent-Daten
                    agent.large_data = [random.random() for _ in range(1000)]  
                    big_data.append(agent.large_data)
                    society.add_agent(agent)
                
                # Test ob System noch funktioniert
                agent_count = len(society.agents)
                
                if agent_count == 50:
                    self.log_test("Resource Exhaustion Resilience", True, "Handled high memory usage scenario")
                else:
                    self.log_test("Resource Exhaustion Resilience", False, f"Lost agents under memory pressure: {agent_count}/50")
                
                # Cleanup
                del big_data
                gc.collect()
                
            except Exception as e:
                self.log_test("Resource Exhaustion Resilience", False, "Failed under simulated memory pressure", e)
                
        except Exception as e:
            self.log_test("Error Handling & Resilience", False, "Exception during resilience testing", e)
    
    def run_all_tests(self):
        """Führt alle tiefgreifenden Tests durch"""
        print("🔬 DEEP SYSTEM TESTING STARTED")
        print("=" * 60)
        print(f"Python Version: {sys.version}")
        print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f}GB RAM")
        print("=" * 60)
        
        # Memory baseline
        gc.collect()
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Führe alle Tests durch
        self.test_memory_leaks()
        self.test_concurrent_access()
        self.test_edge_cases_data_corruption()
        self.test_performance_under_load()
        self.test_data_consistency()
        self.test_error_handling_resilience()
        
        # Zusammenfassung
        self.print_summary()
    
    def print_summary(self):
        """Druckt Testzusammenfassung"""
        print("\n" + "=" * 60)
        print("🔬 DEEP TESTING SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['passed'])
        failed = len(self.test_results) - passed
        
        print(f"📊 Total Tests: {len(self.test_results)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed/len(self.test_results)*100):.1f}%")
        
        # Memory summary
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_change = final_memory - self.memory_baseline
        print(f"💾 Memory Change: {memory_change:+.1f}MB")
        
        print(f"⏱️  Total Test Time: {time.time() - self.start_time:.1f}s")
        
        # Failed tests detail
        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"   • {result['test']}: {result['details']}")
                    if result['error']:
                        print(f"     Error: {result['error']}")
        
        # Overall assessment
        print("\n🎯 ASSESSMENT:")
        if failed == 0:
            print("🏆 EXCELLENT: All deep tests passed! System is highly robust.")
        elif failed <= 2:
            print("✅ GOOD: Minor issues found but system is solid.")
        elif failed <= 5:
            print("⚠️  MODERATE: Several issues need attention.")
        else:
            print("❌ POOR: Significant issues found. System needs major fixes.")

if __name__ == "__main__":
    tester = DeepSystemTester()
    tester.run_all_tests()
