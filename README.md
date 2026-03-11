```markdown
# Projet Deep Learning — Classification multimodale avec CrossViT (O1 → O5)

## 1. Présentation générale du projet

Ce projet explore l’utilisation de **Cross Vision Transformer (CrossViT)** pour une tâche de **classification binaire d’images** en exploitant :
- des **images brutes (non segmentées)**,
- des **images segmentées**,
- et des **informations issues des masques de segmentation** utilisées comme signal spatial explicite (selon l’objectif).

L’objectif est d’évaluer **comment et à quel niveau** l’information de segmentation peut améliorer :
- les performances de classification (Accuracy, F1-score),
- la stabilité de l’apprentissage,
- l’interprétabilité (attention rollout / IoU).

Le travail est structuré en **cinq objectifs expérimentaux (O1 à O5)**, chacun introduisant une modification méthodologique précise.

---

## 2. Contenu du dépôt et arborescence

Structure typique du dépôt :

'''

├── README.md
├── Notebook_Final_DL_Version_GoogleColab.ipynb
├── Notebook_Final_DL_Version_Locale_COMMENTE.ipynb
├── Experiments/
│   ├── O1/
│   ├── O2/
│   ├── O3/
│   ├── O4/
│   └── O5/
│       ├── history.json
│       ├── best.pt
│       ├── last.pt
│       ├── figures/
│       └── summary_tables/
├── herbonaute/ (en fichier .zip)
│   ├── train/
│   ├── 0/
│   └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
├── herbonaute_resized_224/ (en dossier et en fichier .zip)
│   ├── train/
│   ├── 0/
│   └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
└── CrossViT/
└── models/

'''

- `Experiments/` contient l’ensemble des résultats et artefacts produits (courbes, historiques, checkpoints, tableaux).
- `herbonaute/` contient les données prétraitées (voir section suivante).
- `CrossViT/` contient les dépendances modèles nécessaires (si incluses dans le dépôt).

---

## 3. Données

### 3.1 Dataset et prétraitements

Les données proviennent du dataset **Herbonaute**, prétraité pour ce projet :

- Redimensionnement des images en **224 × 224** pixels.
- Organisation par **classe binaire** (`0` / `1`).
- Split **80 % / 20 %** en **train** et **val**.
- Données regroupées dans :

'''

herbonaute/herbonaute_resized_224/

'''

Structure attendue :

'''

herbonaute/herbonaute_resized_224/
├── train/
│   ├── 0/
│   └── 1/
└── val/
├── 0/
└── 1/

'''

### 3.2 Images segmentées et masques

Selon l’objectif, le projet utilise :
- des images non segmentées (`non_seg`),
- des images segmentées (`seg`),
- et/ou des masques dérivés de la segmentation (binarisation/threshold) utilisés comme signal spatial.

---

## 4. Notebooks et environnements d’exécution

Le dépôt contient **deux versions du notebook final**, correspondant à deux environnements.

### 4.1 Version Google Colab

Fichier :
- `Notebook_Final_DL_Version_GoogleColab.ipynb`

Caractéristiques :
- Exécution sur **Google Colab** (GPU).
- Chemins pointant vers un **Google Drive privé partagé en groupe**.
- Installation/usage des dépendances (ex. `timm`) dans l’environnement Colab.
- Version de référence correspondant au pipeline initial.

Usage recommandé :
- Reproduire les expériences dans l’environnement de développement original (Colab + Drive).

### 4.2 Version locale (GPU)

Fichier :
- `Notebook_Final_DL_Version_Locale_COMMENTE.ipynb`

Caractéristiques :
- Exécution sur un **environnement local**, idéalement avec un GPU NVIDIA.
- Chemins pointant vers le **dossier de dépôt local** (pas de Drive).
- Notebook **commenté** : explications méthodologiques, interprétations et commentaires rigoureux (sans modification fonctionnelle du code).

Usage recommandé :
- Exécution finale sur machine locale et rendu académique.

---

## 5. Dépendances et prérequis

### 5.1 Matériel
- GPU recommandé (NVIDIA, CUDA).
- CPU possible mais fortement plus lent.

### 5.2 Python et bibliothèques
- Python 3.9+ recommandé.
- Dépendances principales :
  - `torch`, `torchvision`
  - `timm` (version fixée dans Colab selon le notebook)
  - `numpy`, `pandas`, `matplotlib`
  - `scikit-learn`
  - `tqdm`
  - `Pillow`

Les versions exactes peuvent être confirmées directement dans les cellules “setup” des notebooks.

---

## 6. Objectifs expérimentaux (O1 → O5)

### O1 — Baseline et configurations d’entrée
But : établir une baseline et mesurer l’apport direct de la segmentation.

Configurations évaluées :
- **A** : image non segmentée
- **B** : image segmentée
- **C1 / C2** : double entrée (seg + non-seg) via wrapper DualInput

Sorties :
- métriques (Accuracy, F1)
- historiques (`history.json`)
- checkpoints (`best.pt`, `last.pt`)
- courbes et tableaux récapitulatifs

Conclusion générale (qualitative) :
- la segmentation brute n’apporte pas nécessairement un gain automatique ; la multi-modalité simple ne suffit pas.

---

### O2 — Augmentations géométriques synchronisées
But : préserver l’alignement spatial entre seg et non-seg sous augmentation.

Apport méthodologique :
- augmentations géométriques synchronisées (mêmes transformations sur les deux vues)
- branches mieux contrôlées (réduction de biais structurels)

Conclusion générale (qualitative) :
- amélioration de la rigueur expérimentale, mais la segmentation reste difficile à exploiter sans mécanisme explicite.

---

### O3 — Pondération des patches (Patch-weighting) via masques
But : injecter la segmentation comme **prior spatial explicite** au niveau token/patch, au lieu de remplacer l’image.

Méthode :
- calcul du ratio “plante” par patch (grille ViT)
- pondération des embeddings de patches selon différentes fonctions (linéaire, puissance, saturée, centrée)

Analyses :
- métriques classification
- matrice de confusion, ROC/AUC
- visualisations/heatmaps de focalisation

Conclusion générale (qualitative) :
- la segmentation devient utile lorsqu’elle est injectée au bon niveau (patch/token) comme signal structurant.

---

### O4 — Attention rollout et mesure d’alignement (IoU)
But : évaluer si le modèle se focalise sur les régions pertinentes.

Méthode :
- extraction des cartes d’attention (rollout)
- binarisation et comparaison au masque via IoU

Conclusion générale (qualitative) :
- alignement partiel et instable ; compromis possible entre discrimination (classification) et localisation (cohérence spatiale).

---

### O5 — Combinaison O2 + terme IoU dans la loss
But : guider l’apprentissage en ajoutant l’IoU comme supervision structurée dans la fonction de perte.

Méthode :
- `L_total = L_CE + λ * L_IoU`
- entraînement sur cadre O2 (paires synchronisées)

Conclusion générale (qualitative) :
- amélioration de la cohérence spatiale/interprétabilité, mais optimisation plus instable (multi-objectifs) nécessitant un réglage fin.

---

## 7. Résultats et artefacts (Experiments/)

Tous les résultats sont sauvegardés sous :

'''

Experiments/

'''

Par objectif (O1 à O5) :
- `history.json` : historiques des métriques
- `best.pt`, `last.pt` : checkpoints
- `figures/` : courbes, matrices de confusion, heatmaps, etc.
- `summary_tables/` : tableaux récapitulatifs (si générés)

Ces artefacts servent :
- à l’analyse des performances,
- à la comparaison entre objectifs,
- à la reproductibilité.

---

## 8. Reproductibilité

- Seeds fixés (random, numpy, torch) dans les cellules de setup.
- Les comparaisons entre runs doivent contrôler :
  - mêmes splits train/val,
  - mêmes hyperparamètres,
  - même protocole d’augmentations,
  - même critère de sélection du meilleur modèle (souvent F1(val)).

---

## 9. Exécution rapide

### 9.1 Colab
1. Ouvrir `Notebook_Final_DL_Version_GoogleColab.ipynb`
2. Monter le Drive (cellule prévue)
3. Vérifier les chemins Drive
4. Exécuter de haut en bas

### 9.2 Local (GPU)
1. Ouvrir `Notebook_Final_DL_Version_Locale_COMMENTE.ipynb`
2. Adapter les chemins vers :
   - `herbonaute/herbonaute_resized_224/`
   - `Experiments/`
3. Vérifier l’installation des dépendances (torch + CUDA)
4. Exécuter de haut en bas

---

## 10. Conclusion

La progression O1 → O5 met en évidence un point clé :  
la segmentation n’améliore pas nécessairement la classification si elle est utilisée comme simple entrée.  
Elle devient pertinente lorsqu’elle est intégrée comme **signal spatial structurant** (pondération patch) ou **contrainte d’alignement** (IoU, attention), au prix d’un compromis possible entre stabilité d’entraînement et interprétabilité.
```
#   D e e p L e a r n i n g _ H e r b i e r s - C r o s s V i T _ S e g m e n t a t i o n - - p a t c h - w e i g h t i n g - a n d - i n t e r p r e t a b i l i t y  
 