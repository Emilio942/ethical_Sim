# Technische Aufgabenliste: Migration auf Auditiven Kortex (A1)

Diese Checkliste definiert die notwendigen Schritte, um die bestehende "Visual Expert Module"-Architektur auf eine auditive Verarbeitungspipeline umzustellen.

## Phase 0: Definition der Lernaufgabe (Training Objective)
*Ziel: Damit das Netzwerk sinnvolle STRFs entwickelt, muss es eine Aufgabe lösen.*

- [x] **0.1 Task-Auswahl treffen**
    - [ ] *Option A (Performance-Optimized):* **Speech Command Recognition** (z.B. Google Speech Commands Dataset). Einfach zu trainieren, erzwingt Lernen von spektralen Features.
    - [ ] *Option B (Biologisch Plausibel):* **Self-Supervised Learning** (z.B. Contrastive Predictive Coding). Das Netz lernt, zukünftige Audio-Schnipsel vorherzusagen.
    - [x] *Option C (Emotion):* **Speech Emotion Recognition** (z.B. RAVDESS Dataset). Passt am besten zur "Ethischen Simulation" (Erkennen von Leid/Wut).

- [x] **0.2 Dataset Pipeline aufsetzen**
    - [x] Dataset herunterladen und in das `AudioLoader`-Format integrieren.
    - [x] Train/Test-Split definieren.

## Phase 1: Scanner & Data Ingestion (Auditory Preprocessing)
*Ziel: Ersatz der bildbasierten Pipeline durch biologisch plausible Audio-Verarbeitung.*

- [x] **1.1 Audio-Ingestion-Layer implementieren**
    - [x] Bestehenden `ImageLoader` durch `AudioLoader` ersetzen (Support für WAV/FLAC).
    - [x] Resampling-Routine implementieren (Standardisierung auf z.B. 16kHz oder 44.1kHz).
    - [x] Normalisierung des Audio-Signals (RMS-Level oder dBFS) sicherstellen.

- [x] **1.2 Gammatone-Filterbank implementieren**
    - [x] `scipy.signal` oder `pygammatone` integrieren.
    - [x] Filterbank mit logarithmischer Frequenzverteilung (ERB-Skala) konfigurieren (z.B. 64 oder 128 Kanäle).
    - [x] Frequenzbereich definieren (z.B. 100Hz - 8kHz für Sprache/relevante Signale).

- [x] **1.3 Cochleogramm-Generierung (Feature Extraction)**
    - [x] Hilbert-Transformation auf Filterbank-Outputs anwenden (Einhüllende extrahieren).
    - [x] **Korrektur:** Statt einfacher Log-Kompression eine **PCEN (Per-Channel Energy Normalization)** oder **AGC (Automatic Gain Control)** implementieren.
        *   *Grund:* Biologische Neuronen adaptieren an Hintergrundrauschen. Statische Kompression versagt bei variablen Pegeln (siehe Wang et al., 2017).
    - [x] Downsampling der Einhüllenden auf eine verarbeitbare Zeitauflösung (z.B. 10ms Bins).
    - [x] **Output:** Tensor der Form `(Batch, Channels/Frequencies, TimeSteps)`.

- [x] **1.4 Pre-Emphasis Filter (Engineer's Fix)**
    - [x] Implementierung eines High-Pass Filters zur Kompensation des spektralen Tilts (1/f Spektrum).
    - [x] Integration in `AudioLoader`.

## Phase 2: Strukturelles Mapping (Receptive Fields)
*Ziel: Implementierung von Tonotopie und Spectro-Temporal Receptive Fields (STRFs).*

- [x] **2.1 Retinotopie durch Tonotopie ersetzen**
    - [x] Mapping-Logik umschreiben: Statt `(x, y)` Koordinaten nun `(frequenz, zeit)` Koordinaten.
    - [x] Nachbarschaftsbeziehungen definieren: Neuronen reagieren auf ähnliche Frequenzen (spektrale Nachbarschaft) und zeitliche Muster.

- [x] **2.2 STRF-Basis definieren**
    - [x] Klasse `STRFGenerator` erstellen.
    - [x] Gabor-Filter (2D) durch spectro-temporale Filterkerne ersetzen.
    - [x] Parameter für STRFs definieren (Modulation Power Spectrum - MPS):
        - [x] *Best Frequency (BF)*: Zentrumsfrequenz.
        - [x] *Spectral Modulation (Scale)*: Zyklen pro Oktave (cyc/oct).
        - [x] *Temporal Modulation (Rate)*: Hz (Amplitudenänderung).
        - [x] *Directionality*: Separabilität vs. Inseparabilität (FM-Sweeps).

- [x] **2.3 Input-Mapping auf Experten-Module**
    - [x] Slicing-Logik anpassen: Input-Cochleogramm in überlappende Zeitfenster (Windows) zerlegen.
    - [x] Zuordnung der Input-Kanäle zu den Input-Neuronen der Module basierend auf der Tonotopie.

## Phase 3: Zeitliche Dynamik & Architektur (Expert Model)
*Ziel: Wechsel von statischer Bildverarbeitung zu dynamischer Sequenzverarbeitung.*

- [x] **3.1 Architektur-Anpassung: Asymmetrische 2D-CNNs**
    - [x] *Korrektur:* Keine reine 1D-Conv verwenden, um spektrale Topologie (Tonotopie) zu erhalten (vgl. Kell et al. 2018).
    - [x] Kernel-Größe anpassen: Asymmetrische Filter (z.B. `(Freq=3, Time=5)`).
    - [x] **Dilation:** Implementierung von Dilated Convolutions auf der Zeitachse, um lange Kontexte (>500ms) zu erfassen, ohne die Parameterzahl zu sprengen.## Phase 2: Strukturelles Mapping (Receptive Fields)
*Ziel: Implementierung von Tonotopie und Spectro-Temporal Receptive Fields (STRFs).*

- [ ] **2.1 Retinotopie durch Tonotopie ersetzen**
    - [ ] Mapping-Logik umschreiben: Statt `(x, y)` Koordinaten nun `(frequenz, zeit)` Koordinaten.
    - [ ] Nachbarschaftsbeziehungen definieren: Neuronen reagieren auf ähnliche Frequenzen (spektrale Nachbarschaft) und zeitliche Muster.

- [ ] **2.2 STRF-Basis definieren**
    - [ ] Klasse `STRFGenerator` erstellen.
    - [ ] Gabor-Filter (2D) durch spectro-temporale Filterkerne ersetzen.
    - [ ] Parameter für STRFs definieren (Modulation Power Spectrum - MPS):
        - [ ] *Best Frequency (BF)*: Zentrumsfrequenz.
        - [ ] *Spectral Modulation (Scale)*: Zyklen pro Oktave (cyc/oct).
        - [ ] *Temporal Modulation (Rate)*: Hz (Amplitudenänderung).
        - [ ] *Directionality*: Separabilität vs. Inseparabilität (FM-Sweeps).

- [ ] **2.3 Input-Mapping auf Experten-Module**
    - [ ] Slicing-Logik anpassen: Input-Cochleogramm in überlappende Zeitfenster (Windows) zerlegen.
    - [ ] Zuordnung der Input-Kanäle zu den Input-Neuronen der Module basierend auf der Tonotopie.

## Phase 3: Zeitliche Dynamik & Architektur (Expert Model)
*Ziel: Wechsel von statischer Bildverarbeitung zu dynamischer Sequenzverarbeitung.*

- [ ] **3.1 Architektur-Anpassung: Asymmetrische 2D-CNNs**
    - [ ] *Korrektur:* Keine reine 1D-Conv verwenden, um spektrale Topologie (Tonotopie) zu erhalten (vgl. Kell et al. 2018).
    - [x] Kernel-Größe anpassen: Asymmetrische Filter (z.B. `(Freq=3, Time=5)`).
    - [x] **Dilation:** Implementierung von Dilated Convolutions auf der Zeitachse, um lange Kontexte (>500ms) zu erfassen, ohne die Parameterzahl zu sprengen.

- [ ] **3.2 Recurrent / Attention Mechanismen (Optional)**
    - [ ] LSTM- oder GRU-Layer nach den Convolution-Blöcken einfügen, um Langzeitabhängigkeiten zu modellieren.
    - [ ] Alternativ: Self-Attention Block für globale zeitliche Integration.

- [x] **3.3 Spiking-Dynamik (Falls SNN gewünscht)**
    - [x] Leaky Integrate-and-Fire (LIF) Neuronen-Modell implementieren.
    - [x] Zeitliche Kodierung: Umwandlung der analogen Cochleogramm-Werte in Spike-Trains (Rate Coding oder Latency Coding).

## Phase 4: Validierung & System Identification**
*Ziel: Sicherstellen, dass das Modell wie ein auditiver Kortex reagiert.*

- [x] **4.1 Synthetic Stimuli Tests**
    - [x] Test-Suite mit Pure Tones, White Noise und Frequency Sweeps erstellen.
    - [x] **MTF-Test:** Modulation Transfer Function messen (Reaktion auf amplitudenmoduliertes Rauschen bei verschiedenen Raten).
    - [x] Verifizieren, dass Module frequenzspezifisch (Tuning Curves) reagieren.

- [ ] **4.2 Prediction Performance**
    - [ ] Metrik implementieren: Korrelation zwischen vorhergesagter und tatsächlicher neuronaler Antwort (falls Ground Truth vorhanden) oder Rekonstruktionsfehler.

- [x] **4.3 Sonification & Debugging (Engineer's Fix)**
    - [x] Implementierung von `reconstruct_from_cochleogram` zur akustischen Überprüfung der Features.
    - [x] PCEN-Inversion (Approximation).

## Phase 5: Integration in die Kognitive Architektur
*Ziel: Verbindung des auditiven Kortex mit dem bestehenden Belief-System.*

- [x] **5.1 "Perception-to-Concept" Interface**
    - [x] Implementierung eines **Readout-Layers** (Global Average Pooling + Dense Layer), der die hochdimensionalen A1-Features auf abstrakte Konzepte mappt.
    - [x] Beispiel: Mapping von A1-Aktivität -> "Detected Emotion: Anger" -> Input für `NeuralEthicalAgent`.

- [x] **5.2 Echtzeit-Loop Integration**
    - [x] Sicherstellen, dass der Audio-Stream asynchron verarbeitet wird, um die Entscheidungs-Loop der Agenten nicht zu blockieren.
    - [x] Integration in `NeuralEthicalAgent`: Methoden `initialize_auditory_system` und `perceive_audio` hinzugefügt.

- [ ] **5.3 Attention-Based Readout (Brain-IT Inspired)**
    - [ ] *Konzept:* Statt einfachem Pooling einen Attention-Mechanismus implementieren, der lernt, welche spektralen/zeitlichen Features für eine bestimmte Emotion relevant sind.
    - [ ] Implementierung eines kleinen Transformer-Encoder-Blocks oder Attention-Heads als Schnittstelle zwischen A1 und Agent.




