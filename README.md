```markdown
# Projet Deep Learning — Classification multimodale avec CrossViT (O1 → O5)

## 1. Présentation générale du projet

Ce projet explore l’utilisation de **Cross Vision Transformer (CrossViT)** pour une tâche de **classification binaire d’images** en exploitant :

- des **images brutes (non segmentées)**,
- des **images segmentées**,
- des **informations issues des masques de segmentation** utilisées comme signal spatial explicite (selon l’objectif).

L’objectif est d’évaluer **comment et à quel niveau** l’information de segmentation peut améliorer :

- les performances de classification (Accuracy, F1-score),
- la stabilité de l’apprentissage,
- l’interprétabilité (attention rollout / IoU).

Le travail est structuré en **cinq objectifs expérimentaux (O1 à O5)**, chacun introduisant une modification méthodologique précise.

---

## 2. Contenu du dépôt et arborescence

Structure typique du dépôt :

```

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
├── herbonaute/ (archive .zip)
│   ├── train/
│   │   ├── 0/
│   │   └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
├── herbonaute_resized_224/ (dossier ou archive .zip)
│   ├── train/
│   │   ├── 0/
│   │   └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
└── CrossViT/
└── models/

```

- `Experiments/` contient l’ensemble des résultats et artefacts produits (courbes, historiques, checkpoints, tableaux).
- `herbonaute/` contient les données prétraitées (voir section suivante).
- `CrossViT/` contient les dépendances modèles nécessaires (si incluses dans le dépôt).

---

## 3. Données

⚠️ *Les données ne sont pas incluses dans ce dépôt en raison de la confidentialité du projet de recherche dont elles sont issues.*

### 3.1 Dataset et prétraitements

Les données proviennent du dataset **Herbonaute**, prétraité pour ce projet :

- redimensionnement des images en **224 × 224 pixels** ;
- organisation par **classe binaire** (`0` / `1`) ;
- split **80 % / 20 %** en **train** et **val**.

Les données sont organisées comme suit :

```

herbonaute/herbonaute_resized_224/
├── train/
│   ├── 0/
│   └── 1/
└── val/
├── 0/
└── 1/

```

### 3.2 Images segmentées et masques

Selon l’objectif expérimental, le projet utilise :

- des images non segmentées (`non_seg`),
- des images segmentées (`seg`),
- et/ou des **masques dérivés de la segmentation** (binarisation / threshold) utilisés comme **signal spatial explicite**.

---

## 4. Notebooks et environnements d’exécution

Le dépôt contient **deux versions du notebook final**, correspondant à deux environnements d’exécution.

### 4.1 Version Google Colab

Fichier :

`Notebook_Final_DL_Version_GoogleColab.ipynb`

Caractéristiques :

- exécution sur **Google Colab (GPU)** ;
- chemins pointant vers un **Google Drive privé partagé** ;
- installation des dépendances (ex. `timm`) directement dans l’environnement Colab ;
- version correspondant au **pipeline de développement initial**.

Usage recommandé :

reproduire les expériences dans l’environnement original (**Colab + Drive**).

---

### 4.2 Version locale (GPU)

Fichier :

`Notebook_Final_DL_Version_Locale_COMMENTE.ipynb`

Caractéristiques :

- exécution sur **un environnement local**, idéalement avec un **GPU NVIDIA** ;
- chemins pointant vers le **dossier local du dépôt** ;
- notebook **commenté**, incluant des explications méthodologiques et des interprétations des résultats.

Usage recommandé :

exécution finale sur **machine locale** et utilisation dans un contexte académique.

---

## 5. Dépendances et prérequis

### 5.1 Matériel

- GPU recommandé (NVIDIA + CUDA)
- CPU possible mais significativement plus lent

### 5.2 Python et bibliothèques

- Python **3.9+** recommandé

Principales dépendances :

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`
- `Pillow`

Les versions exactes peuvent être vérifiées dans les cellules **setup** des notebooks.

---

## 6. Objectifs expérimentaux (O1 → O5)

### O1 — Baseline et configurations d’entrée

**But :** établir une baseline et mesurer l’apport direct de la segmentation.

Configurations évaluées :

- **A** : image non segmentée  
- **B** : image segmentée  
- **C1 / C2** : double entrée (seg + non-seg) via wrapper DualInput

Sorties :

- métriques (Accuracy, F1)
- historiques (`history.json`)
- checkpoints (`best.pt`, `last.pt`)
- courbes et tableaux récapitulatifs

Conclusion générale :

la segmentation brute n’apporte pas nécessairement un gain automatique ; une simple multimodalité ne suffit pas.

---

### O2 — Augmentations géométriques synchronisées

**But :** préserver l’alignement spatial entre les images segmentées et non segmentées sous augmentation.

Apport méthodologique :

- augmentations géométriques synchronisées ;
- branches d’entrée mieux contrôlées.

Conclusion :

amélioration de la rigueur expérimentale, mais la segmentation reste difficile à exploiter sans mécanisme explicite.

---

### O3 — Pondération des patches (Patch Weighting)

**But :** injecter la segmentation comme **prior spatial explicite** au niveau des **tokens / patches**.

Méthode :

- calcul du ratio *plante* par patch (grille ViT) ;
- pondération des embeddings selon différentes fonctions.

Analyses :

- métriques de classification
- matrice de confusion
- ROC / AUC
- visualisations (heatmaps)

Conclusion :

la segmentation devient utile lorsqu’elle est injectée **au bon niveau d’abstraction (patch/token)**.

---

### O4 — Attention rollout et alignement spatial

**But :** analyser si le modèle se focalise sur les régions pertinentes.

Méthode :

- extraction des cartes d’attention (rollout) ;
- comparaison aux masques via **IoU**.

Conclusion :

alignement partiel et instable entre **discrimination (classification)** et **localisation spatiale**.

---

### O5 — Combinaison O2 + terme IoU dans la loss

**But :** guider l’apprentissage via une **supervision spatiale explicite**.

Fonction de perte :

```

L_total = L_CE + λ · L_IoU

```

Conclusion :

meilleure cohérence spatiale et interprétabilité, mais **optimisation plus instable** (problème multi-objectifs).

---

## 7. Résultats et artefacts

Tous les résultats sont sauvegardés dans :

```

Experiments/

```

Par objectif :

- `history.json` — historiques d’entraînement
- `best.pt`, `last.pt` — checkpoints
- `figures/` — courbes, matrices de confusion, heatmaps
- `summary_tables/` — tableaux récapitulatifs

Ces artefacts permettent :

- l’analyse des performances,
- la comparaison entre objectifs,
- la reproductibilité.

---

## 8. Reproductibilité

- Seeds fixées (`random`, `numpy`, `torch`)
- Comparaisons contrôlées selon :
  - mêmes splits train/val
  - mêmes hyperparamètres
  - même protocole d’augmentation
  - même critère de sélection du meilleur modèle (souvent **F1(val)**)

---

## 9. Exécution rapide

### Colab

1. Ouvrir `Notebook_Final_DL_Version_GoogleColab.ipynb`
2. Monter le Drive
3. Vérifier les chemins
4. Exécuter les cellules

### Local (GPU)

1. Ouvrir `Notebook_Final_DL_Version_Locale_COMMENTE.ipynb`
2. Adapter les chemins vers :
   - `herbonaute/herbonaute_resized_224/`
   - `Experiments/`
3. Vérifier `torch + CUDA`
4. Exécuter les cellules

---

## 10. Conclusion

La progression **O1 → O5** montre que la segmentation n’améliore pas nécessairement la classification lorsqu’elle est utilisée comme **simple entrée visuelle**.

Elle devient réellement pertinente lorsqu’elle est intégrée comme :

- **signal spatial structurant** (pondération des patches),
- **contrainte d’alignement** (IoU, attention).

Cette approche améliore l’interprétabilité mais introduit un compromis potentiel entre **stabilité d’entraînement** et **richesse du signal spatial**.
```
