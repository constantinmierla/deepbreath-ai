
# 🧠 Sistem Inteligent pentru Diagnosticarea Timpurie a Cancerului Pulmonar

## 📌 Introducere

Cancerul pulmonar este una dintre cele mai agresive forme de cancer, având o rată mare de mortalitate. Detectarea timpurie joacă un rol esențial în șansele de supraviețuire ale pacienților, însă metodele tradiționale de diagnostic bazate pe interpretarea umană a imaginilor CT sunt consumatoare de timp și pot fi predispuse la erori.

Acest proiect își propune să construiască un sistem de **inteligență artificială** care analizează imagini CT toracice pentru a:
- Detecta prezența cancerului pulmonar;
- Clasifica tipul de cancer;
- Sprijini medicii în luarea deciziilor clinice rapide și precise.

---

## ✅ Etapa 1: Definirea Problemei

### 🔍 Ce se dă?
- Imagini CT în format `.jpg` sau `.png`, grupate în patru categorii:
  - Adenocarcinom
  - Carcinom cu celule mari (Large-cell)
  - Carcinom scuamos (Squamous-cell)
  - Normal (celule sănătoase)
- Structura datasetului:
  - `train/`: 70% din imagini
  - `valid/`: 10%
  - `test/`: 20%

### 🎯 Ce se cere?
- Construirea unui model AI (bazat pe CNN) care:
  - Recunoaște dacă pacientul are sau nu cancer pulmonar;
  - Clasifică tipul de cancer în una dintre cele 3 categorii;
  - Poate fi folosit în aplicații clinice pentru sprijin decizional.

### 🤖 De ce e nevoie de AI?
- Volumul mare de imagini → imposibil de analizat rapid de oameni;
- Modelele CNN pot învăța tipare subtile și au performanță superioară în clasificare de imagini;
- AI-ul poate standardiza diagnosticul, reducând erorile și timpul de interpretare.

---

## 🔎 Etapa 2: Analiza Datelor de Intrare

### 📁 Descrierea Datasetului
- **Sursă**: [Chest CT-Scan images Dataset – Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data)
- **Format**: `.jpg`, `.png`
- **Număr total de imagini**: mii de imagini etichetate pe categorii (număr exact vizibil după descărcare locală)
- **Scopul inițial**: creare de model AI care să clasifice automat tipul de cancer pulmonar

### 🩺 Tipurile de cancer din setul de date:

#### 🔬 Adenocarcinom
- Cel mai frecvent tip de cancer pulmonar (30-40% din cazuri de NSCLC).
- Apare în glandele care secretă mucus în zonele periferice ale plămânilor.
- Simptome: tuse, răgușeală, slăbiciune, scădere în greutate.

#### 🔬 Carcinom cu celule mari (Large-cell)
- Apare oriunde în plămân.
- Se dezvoltă rapid și este responsabil pentru 10-15% din NSCLC.
- Este un tip agresiv și greu de tratat.

#### 🔬 Carcinom scuamos (Squamous-cell)
- Apare central, unde bronhiile mari se conectează la trahee.
- Responsabil pentru 30% din cazurile de NSCLC.
- Puternic asociat cu fumatul.

#### ✅ Celule normale (Normal)
- Imagini CT fără anomalii vizibile.
- Esențiale pentru antrenarea corectă a modelului AI (contrast între „normal” și „patologic”).

---

## 🗂️ Structura Dataset-ului

| Folder   | Descriere                  | Procent din set |
|----------|----------------------------|------------------|
| `train/` | Imagini pentru antrenare   | 70%              |
| `valid/` | Imagini pentru validare    | 10%              |
| `test/`  | Imagini pentru testare     | 20%              |

---


