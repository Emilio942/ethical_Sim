# Formelsammlung des Ethik-Projekts

Dieses Dokument listet die zentralen Formeln auf, die im Ethik-Simulationsprojekt verwendet werden.

## Agenten-Modell (`src/agents/agents.py`)

### 1. Kognitive Dissonanz

Die kognitive Dissonanz wird als Summe der Konflikte zwischen verbundenen Überzeugungen berechnet.

```
Dissonanz = Σ (Stärke_A * Stärke_B * Einfluss_AB * |Polarität_AB| * Gewissheit_A * Gewissheit_B)
```

- **Stärke_A, Stärke_B**: Stärke der Überzeugungen A und B.
- **Einfluss_AB**: Stärke der Verbindung zwischen A und B.
- **Polarität_AB**: Art der Verbindung (negativ für Konflikt).
- **Gewissheit_A, Gewissheit_B**: Gewissheit der Überzeugungen.

### 2. Bayes'sches Update der Überzeugungen

Die Stärke einer Überzeugung wird basierend auf neuen "Beweisen" (Erfahrungen) aktualisiert.

```
neue_Stärke = apply_bayesian_update(alte_Stärke, Stärke_Beweis, Richtung_Beweis)
```

Die genaue Formel befindet sich in `cognitive_architecture.py`.

### 3. Q-Learning (Bestärkendes Lernen)

Der Wert einer Aktion in einem bestimmten Szenario-Typ wird über die Zeit gelernt.

```
neuer_Wert = alter_Wert + Lernrate * (Belohnung - alter_Wert)
```

- **Lernrate**: Wie schnell der Agent lernt.
- **Belohnung**: Das Ergebnis der Aktion.

### 4. Bedauern-Minimierung (Regret-Minimization)

Eine "Strafe" für Entscheidungen, die potenziell zu hohem Bedauern führen könnten.

```
Bedauern_Strafe = potenzielles_Bedauern * 0.2
```

### 5. Spreading Activation (Aktivierungsausbreitung)

Die Aktivierung einer Überzeugung breitet sich auf benachbarte Überzeugungen aus.

```
Ausbreitungsaktivierung = Aktivierung_Quelle * Verbindungsstärke * 0.5
```

### 6. Ausbreitung von Überzeugungsänderungen

Wenn sich eine Überzeugung ändert, wird diese Änderung an verbundene Überzeugungen weitergegeben.

```
Änderung_B = Änderung_A * Einfluss_AB * Polarität_AB * Ausbreitungsstärke
```

### 7. Sozialer Einfluss

Die Überzeugung eines Agenten wird durch die eines anderen Agenten beeinflusst.

```
Änderung = Differenz * Lernrate * Faktor_Eigene_Gewissheit * Faktor_Andere_Gewissheit * Gruppenfaktor
```

- **Differenz**: Unterschied in der Überzeugungsstärke.
- **Faktoren**: Modulatoren basierend auf Gewissheit und Gruppenzugehörigkeit.

## Metriken (`src/analysis/metrics.py`)

### 1. Interne Konsistenz

Misst die Widerspruchsfreiheit der Überzeugungen eines Agenten.

```
Konsistenz = 1.0 / (1.0 + Varianz_der_Überzeugungsstärken)
```

### 2. Zeitliche Stabilität

Misst, wie stabil die Entscheidungen eines Agenten über die Zeit sind.

```
Stabilität = 1.0 / (1.0 + Standardabweichung_der_Entscheidungen)
```

### 3. Polarisationsindex

Misst die durchschnittliche "Distanz" zwischen den Überzeugungen aller Agenten in der Gesellschaft.

```
Polarisation = Durchschnittliche_Distanz / 2.0
```

### 4. Konsens-Level

Misst den Anteil der Agentenpaare, deren Überzeugungen sehr ähnlich sind.

```
Konsens = Anzahl_Konsenspaare / Gesamtanzahl_Paare
```

### 5. Netzwerk-Kohäsion (Dichte)

Misst, wie stark das soziale Netzwerk verknüpft ist.

```
Dichte = Anzahl_Verbindungen / Maximal_mögliche_Verbindungen
```

### 6. Gini-Koeffizient

Wird verwendet, um die Ungleichheit in der Verteilung von Einfluss zu messen.

```
Gini = (n + 1 - 2 * Σ(kumulative_Summe) / letzte_kumulative_Summe) / n
```

### 7. Überzeugungsdistanz (Euklidisch)

Der Abstand zwischen den Überzeugungsvektoren zweier Agenten.

```
Distanz = sqrt(Σ (Stärke_A_i - Stärke_B_i)^2 / Anzahl_gemeinsamer_Überzeugungen)
```

### 8. Entscheidungsdiversität (Entropie)

Misst die Vielfalt in den Entscheidungen eines Agenten.

```
Entropie = -Σ (p_i * log2(p_i))
```
- **p_i**: Wahrscheinlichkeit der Entscheidung i.

### 9. Ethische Ausrichtung

Wie gut die Überzeugungen eines Agenten mit einem Zielsatz von Prinzipien übereinstimmen.

```
Ausrichtung = 1.0 - |Wert_Agent - Zielwert| / 2.0
```

## Kognitive Kernkomponenten (`src/core/`)

### 1. Aktivierungsabfall (`beliefs.py`)

Die Aktivierung einer Überzeugung nimmt über die Zeit exponentiell ab.

```
Abfallfaktor = exp(-0.1 * Zeit_seit_letzter_Aktivierung)
```

### 2. Vereinfachtes Bayes'sches Update (`cognitive_architecture.py`)

Eine vereinfachte Formel zur Aktualisierung von Überzeugungen.

```
Änderung = Beweisstärke * (1.0 - alte_Überzeugung) * Updaterate
neue_Überzeugung = alte_Überzeugung + Änderung + Anker-Effekt
```