
>***** Ce projet a été réalisé dans le cadre du Master Informatique & Télécommunications TAM
         par Hajar Aiz et Mohamed Jabraoui sous l'encadrement de Mr Abdelhak Mahmoudi *****

>
          # DL Models Optimazation : Pruning, Quantization et Knowledge Distillation   #

## Objectif du Projet

Ce projet a pour objectif d’explorer et d’appliquer trois techniques majeures d’optimisation des modèles Deep Learning :

- **Pruning (Élagage)** : Réduction du nombre de poids d’un modèle en supprimant les connexions les moins importantes.
- **Quantization (Quantification)** : Réduction de la précision des poids (ex. : passer de 32 bits à 8 bits) pour alléger le modèle.
- **Knowledge Distillation (Distillation des Connaissances)** : Transfert des connaissances d’un modèle complexe (Teacher) vers un modèle plus léger (Student).

L’objectif final est de **réduire la taille des modèles et accélérer leur exécution tout en maintenant une précision acceptable**.
 
##  Environnement requis :

Voici les bibliothèques nécessaires pour exécuter le projet :

```bash
pip install torch torchvision
```

- Python ≥ 3.0
- PyTorch ≥ 1.12
- torchvision = 2.1.0
- Google Colab recommandé pour bénéficier du GPU 

---
##  Structure du projet  :

| Fichier                   | Description                                             |
|---------------------------|---------------------------------------------------------|
| `DL_Optimization.ipynb`   | Notebook principal contenant l’ensemble des expériences |
| `README.md`               | Ce fichier, expliquant le projet en détail              |
| `Vidéo dimenstrative`     | expliquant le code et les résultats                     |

---

## Lancer le projet :

1. Ouvrir le notebook `DL_Optimization.ipynb` dans Google Colab.
2. Exécuter chaque cellule en suivant l’ordre :
   - Pruning
   - Quantization
   - Knowledge Distillation
3. Observer les tailles des modèles, les pertes et la similarité des prédictions.

---


##  Méthodologie

###  1. Pruning

- Modèle utilisé : `ResNet18`
- Pruning appliqué sur toutes les couches `Conv2D` et `Linear` (90% des poids supprimés)
- Suppression des poids inutiles avec `prune.remove`
- Comparaison de la **taille du modèle avant et après pruning** (avec compression `.gz`)

###  2. Quantization

- Quantification statique appliquée au modèle ResNet18
- Calibration avec des données aléatoires (dummy data)
- Conversion du modèle au format `int8`
- Mesure de la **réduction de taille** du modèle quantifié

###  3. Knowledge Distillation

- Teacher : `ResNet18` (complexe)
- Student : `MobileNetV2` (léger)
- Données : Batch d’images factices (dummy)
- Loss combinée : `KLDivLoss` + `CrossEntropy`
- Entraînement sur plusieurs epochs
- Évaluation par **taux de similarité des prédictions**

---

##  Résultats attendus

| Technique               | Objectif principal                  | Résultat observé                       |
|-------------------------|-------------------------------------|----------------------------------------|
| Pruning                 | Réduction du nombre de poids        | Taille réduite (avec gzip)             |
| Quantization            | Réduction de la précision           | Taille réduite + exécution plus rapide |
| Knowledge Distillation  | Réduction du modèle avec précision  | Student plus rapide mais moins précis  |

>  Le pruning seul ne réduit pas la taille du fichier `.pth`, car les zéros restent présents. Il faut compresser ou utiliser des formats spécialisés (comme ONNX ou TorchScript) pour un vrai gain de taille.

##  Problèmes connus

-  **CUDA Out Of Memory** : le projet peut dépasser la mémoire GPU. Solution : forcer l’exécution sur CPU ou réduire la taille des batchs.
-  **Incohérence CPU/GPU** : attention à bien envoyer les données et modèles sur le même appareil (`cpu` ou `cuda`).

---

##  Comment exécuter le projet

1. Ouvre le fichier `DL_Optimization.ipynb` dans **Google Colab**
2. Utiliser T4 GPU comme processeur
2. Exécute les cellules une à une (pruning, quantization, distillation)
3. Les modèles compressés ou quantifiés seront sauvegardés automatiquement
4. Tu peux visualiser les résultats dans les impressions (`print()`)

---

##  Références

- PyTorch Pruning: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- Quantization: https://pytorch.org/docs/stable/quantization.html
- Distillation: Hinton et al., 2015 – "Distilling the Knowledge in a Neural Network"
- Deep Learning Book
- Dive Into DL Book
---

## Conclusion

Ce projet montre comment les approches complémentaires — **pruning**, **quantization** et **distillation** — permettent d’y parvenir.
L'optimisation des modèles est essentielle pour déployer l’IA dans des environnements à ressources limitées (mobile, embarqué).  

---
