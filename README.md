# A logical model of signed-XOR learning

Code and preprint for **"A logical model of signed-XOR learning: Three-valued Łukasiewicz logic exposes the structure of a local neuronal learning rule"** (J. Marco de Lucas, IFCA–CSIC).

The signed-XOR learning rule is a local rule proposed as a biological analogue of a loss function (Peña, Lloret & Marco, 2024, 2026). Its signals have an exact representation in three-valued Łukasiewicz logic Ł₃ on {−1, 0, 1} (inhibition, silence, excitation), extended with Słupecki's constant T = 0. This repository contains:

- the Ł₃+T algebra and an exhaustive search for minimal formulas;
- truth-table verification of every identity stated in the paper;
- a hybrid autoencoder (integer integration + Ł₃ signals) trained on MNIST with the signed-XOR rule, in three variants.

## Main formal results

| Signal | Minimal Ł₃+T formula | Needs T? |
|---|---|---|
| Signed error e = clip(x − f) | `(x → f) → (T ⊙ (f → x))` | yes (Proposition 1) |
| Signed Hebbian term h·e | `(h ⊕ e) → (h ∧ e)` | no |

With the online decoder update ΔV = η hᵀe, every erroneous output bit moves its potential in the direction of e. This holds for binary and ternary hidden layers (Proposition 3).

## Repository structure

```
l3.py        Ł3+T operations, formula search, truth-table verification
ae.py        autoencoder: variants `paper`, `l3bin`, `l3ter`
seeds.py     run one variant with a given seed
paper/       preprint (LaTeX source and PDF)
results/     MNIST results, three seeds per variant (JSON)
```

## Requirements

Python ≥ 3.9 and NumPy.

```bash
pip install -r requirements.txt
```

## Usage

**Verify the identities** (a few milliseconds):

```bash
python3 l3.py
# identidades verificadas: True
```

**Rerun the minimal-formula search:**

```bash
python3 -c "
import l3, numpy as np
t = {'error': l3.truth_table(lambda x,y: np.clip(x-y,-1,1)),
     'product': l3.truth_table(lambda x,y: x*y)}
print('with T:   ', l3.search(t, 11, True))
print('without T:', l3.search(t, 11, False))
"
```

Without T, only the product formula is found. This agrees with Proposition 1, which excludes T-free formulas for the error at any size.

**Reproduce the MNIST experiments:**

```bash
bash fetch_mnist.sh                 # downloads MNIST into data/
python3 ae.py paper l3bin l3ter     # seed 0, all three variants
python3 seeds.py l3ter 1            # one variant, a given seed
```

Each variant takes about a minute per run on one CPU core. On Windows, run the download commands from `fetch_mnist.sh` manually or use Git Bash. If Ł₃ symbols print incorrectly, run `chcp 65001` first.

## Variants

| Variant | Hidden layer | Weights | Signals |
|---|---|---|---|
| `paper` | {0,1} | real-valued | arithmetic (replica of the original rule) |
| `l3bin` | {0,1} | integer | Ł₃+T formulas |
| `l3ter` | {−1,0,1}, dead zone θ = 0.5·median(\|u\|) | integer | Ł₃+T formulas |

## Results (MNIST test set, 5 epochs, mean ± s.d. over 3 seeds)

| Variant | Bit-wise acc. (%) | Balanced acc. (%) | F1 | Linear probe (%) |
|---|---|---|---|---|
| `paper` | 95.72 ± 0.10 | 90.59 ± 1.03 | 0.840 ± 0.007 | 90.6 ± 1.2 |
| `l3bin` | 95.68 ± 0.38 | 90.55 ± 1.28 | 0.838 ± 0.016 | 90.9 ± 0.5 |
| `l3ter` | 95.70 ± 0.23 | 91.24 ± 0.48 | 0.842 ± 0.007 | 89.3 ± 2.2 |

The equivalence between logical and arithmetic signals is algebraic. The experiments show that the resulting hybrid, integer implementation works and is comparable to a real-valued replica. The replica is based on the text of the original preprint and falls about two points below its published figures.

## Related work

- M. Peña Fernández, L. Lloret Iglesias, J. Marco de Lucas. *Toward defining loss functions in neuroscience: an XOR-based neuronal mechanism.* bioRxiv, 2026. doi:10.64898/2026.03.16.712061
- M. Peña, J. Marco, L. Lloret. *Implementing engrams from a machine learning perspective: XOR as a basic motif.* arXiv:2406.09940, 2024.

## Citation

```bibtex
@misc{marco2026logicalxor,
  author = {Marco de Lucas, Jes{\'u}s},
  title  = {A logical model of signed-{XOR} learning: Three-valued {\L}ukasiewicz logic exposes the structure of a local neuronal learning rule},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgements

Supported by the ENGRAMMER project (IASOMM2024007), funded by EU Next Generation under the Recovery and Resilience Facility (RRF).
