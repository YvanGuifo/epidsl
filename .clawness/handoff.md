# Reprise — EpiDSL

**Où on en est.** Branche `main`, arbre de travail modifié (non committé). Dernier
commit `7e30045`. 334 tests verts. La tranche « bruit de mesure » est terminée et
prête à commiter ; attente de la validation de l'utilisateur.

**Tranche terminée : bruit de mesure.**
- Syntaxe `observe X = expr ~ poisson()` / `~ normal(sd)` / `~ negbinomial(phi)`.
- NLL dans `calibration.py::distance()` pour les observables avec bruit.
- Tirage stochastique dans `backends/__init__.py::observe()` via `rng=rng` (Gillespie et tau-leap).
- Vérification C13 sur les paramètres de bruit.
- 13 tests dans `tests/test_noise.py` (parsing, C13, calibration ×3, stochastique ×4).
- Exemple `examples/sir_noise.epidsl`, D23, CHANGELOG, README.

**Prochaine action :** demander à l'utilisateur de commiter avec
`git commit -m "..."` ou dire « Validez ».

**Restant.** Backend multi-agents sur réseau (syntaxe à choisir), serveur de langage,
vérifications C9/C10, inversion plus rapide pour le Gillespie en temps variable (D22),
épingler les actions de `.github/workflows/ci.yml` par SHA, dépôt distant et
publication PyPI.

**Questions ouvertes.** Aucune.

**Autostart:** yes
