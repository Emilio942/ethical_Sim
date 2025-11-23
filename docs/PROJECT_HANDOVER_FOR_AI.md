# Project Handover: Ethical Simulation & Auditory Cortex
**Date:** November 23, 2025
**Author:** GitHub Copilot (Gemini 3 Pro)
**To:** Future AI Assistant / Developer

## 1. Project Mission
This project aims to build a **Neural Ethical Agent** capable of:
1.  **Social Interaction:** Simulating complex societies with beliefs, morals, and group dynamics.
2.  **Auditory Perception:** "Hearing" emotional speech using a biologically plausible **Auditory Cortex** (not just a black-box CNN).
3.  **Ethical Decision Making:** Resolving dilemmas (Trolley Problem, etc.) based on internal belief networks.

## 2. Architecture Overview
The system is divided into three "Brains":

### A. The Cognitive Brain (`src/core/cognitive_architecture.py`)
- Manages beliefs (`NeuralEthicalBelief`), personality (Big Five), and decision making.
- Uses a "Global Workspace Theory" inspired approach.

### B. The Auditory Brain (`src/core/auditory_cortex.py`)
- **Input:** Raw Audio (WAV).
- **Processing:**
    1.  **Cochlea:** Gammatone Filters + PCEN (Simulates ear adaptation).
    2.  **A1 Cortex:** Spectro-Temporal Receptive Fields (STRFs) with Dilated Convolutions.
    3.  **Readout:** MLP Classifier (currently) mapping cortical spikes to emotions.
- **Status:** Functional. Validated on EMO-DB (58% accuracy).

### C. The Social Brain (`src/society/`)
- Manages interactions between multiple `NeuralEthicalAgent` instances.
- Visualized via a 3D Plotly network.

## 3. Key Files & Locations
- **Entry Point:** `main.py` (Web Server + Simulation Loop).
- **Agents:** `src/agents/neural_agent.py` (The main class).
- **Auditory Core:** `src/core/auditory_cortex.py` (The biological model).
- **Tools:** `tools/generate_synthetic_data.py` (Data augmentation).
- **Demos:** `demos/` (Validation scripts).
- **Docs:** `docs/` (Detailed status reports).

## 4. Current Status & Known Issues
**✅ Working:**
- Full Web UI with Live Dashboard.
- Auditory Pipeline (Signal Processing -> Classification).
- Agent "Hearing" Integration.
- Data Augmentation Tool (`tools/generate_synthetic_data.py`).

**⚠️ Issues (The "Engineering Gap"):**
1.  **Data Scarcity:** The model is trained on EMO-DB (535 files). It needs 10k+ files to be robust. Use the `generate_synthetic_data.py` tool to help, but real data is better.
2.  **Temporal Readout:** The current MLP averages features over time. A Recurrent Neural Network (LSTM/GRU) would be much better for detecting changing emotions.
3.  **Performance:** The STRF convolution is CPU-heavy. GPU acceleration (PyTorch) is implemented for parts, but could be optimized.

## 5. Instructions for the Next AI
1.  **Start Here:** Run `demos/demo_agent_integration.py` to verify the system is alive.
2.  **Fix the Data:** If the user provides "Gold Data", use `tools/generate_synthetic_data.py` to multiply it.
3.  **Upgrade the Brain:** Replace the `MLPClassifier` in `demos/demo_full_pipeline.py` with a PyTorch LSTM.
4.  **Connect the Loop:** Make the agent *say* something back (Text-to-Speech) based on what it heard.

*Good luck. The architecture is solid, it just needs scale.*
