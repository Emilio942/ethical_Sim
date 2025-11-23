# Empfohlene Datensätze für die Auditive Migration

Basierend auf der Recherche auf Hugging Face und dem Projektkontext ("Ethik-Simulation", deutsche Kommentare) sind hier die besten Kandidaten für das Training des Auditiven Kortex.

## 1. Emotion Recognition (Priorität für Ethik-Simulation)

Diese Datensätze eignen sich hervorragend für **Phase 0.1 Option C**, da sie Emotionen wie Wut, Angst und Trauer enthalten – essenziell für einen ethischen Agenten, der Leid erkennen soll.

### **A. EMO-DB (Berlin Database of Emotional Speech)**
*   **Hugging Face ID:** `renumics/emodb`
*   **Sprache:** Deutsch 🇩🇪
*   **Beschreibung:** Der Standard-Datensatz für deutsche Emotionserkennung. 10 Schauspieler (5m/5w) sprechen 10 Sätze in verschiedenen Emotionen.
*   **Emotionen:** Wut, Langeweile, Ekel, Angst, Freude, Trauer, Neutral.
*   **Größe:** 535 Aufnahmen.
*   **Warum hier?** Da das Projekt eine deutsche Struktur hat, ist dies der **natürlichste Startpunkt**.

### **B. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
*   **Hugging Face ID:** `TwinkStart/RAVDESS` (oder `viks66/ravdess_speech`)
*   **Sprache:** Englisch 🇺🇸
*   **Beschreibung:** Sehr sauberer, professioneller Datensatz. Validierte emotionale Intensität.
*   **Emotionen:** Neutral, Ruhig, Glücklich, Traurig, Wütend, Ängstlich, Ekel, Überrascht.
*   **Größe:** ~1440 Sprach-Dateien.
*   **Warum hier?** Hohe Audioqualität, gut für sauberes Training der STRFs.

### **C. Combined Dataset (RAVDESS + CREMA-D + TESS + SAVEE)**
*   **Hugging Face ID:** `stapesai/ssi-speech-emotion-recognition`
*   **Sprache:** Englisch
*   **Beschreibung:** Eine Zusammenfassung der vier wichtigsten englischen Datensätze.
*   **Größe:** ~12.000 Samples.
*   **Warum hier?** Wenn das Modell robust werden soll (Generalisierung), ist dies die beste Wahl.

---

## 2. Speech Commands (Für technische Validierung)

Geeignet für **Phase 0.1 Option A** (Performance-Optimierung), um sicherzustellen, dass die Architektur überhaupt lernt.

### **Google Speech Commands (v0.02)**
*   **Hugging Face ID:** `google/speech_commands`
*   **Beschreibung:** Ein-Sekunden-Schnipsel von Befehlen ("Yes", "No", "Stop", "Go").
*   **Größe:** >100.000 Samples.
*   **Warum hier?** Perfekt zum Debuggen der *Dilated Convolutions*, da die zeitliche Struktur kurz und klar ist.

---

## Empfehlung für das weitere Vorgehen

1.  **Installation:** Wir benötigen die `datasets` Library von Hugging Face.
    ```bash
    pip install datasets librosa
    ```
2.  **Start:** Ich empfehle, mit **EMO-DB** zu beginnen, da es klein, überschaubar und deutschsprachig ist. Das ermöglicht schnelle Iterationen beim Testen der Pipeline.
