# ToyARK13

Hughes/Sudderth's autoregressive toy dataset (x-hdphmm-nips2015): 13 true AR
regimes, 52 sequences x 800 steps, 3-D observations, true self-persistence
0.989. `HMMdataset.mat` fields: `X` (T_total, 3), `TrueZ`, `doc_range`.

Reference: Hughes, Stephenson & Sudderth, "Scalable adaptation of state
complexity for nonparametric hidden Markov models", NIPS 2015.

## Contents
- `HMMdataset.mat`          the dataset
- `run_hughes_protocol.py`  reproduces the Hughes protocol trace from the
                            top-level README (K=25 start, randcontigblocks,
                            merge/delete/seqcreate moves on a frozen corpus;
                            `NSEQ=12 LAPS=30 python run_hughes_protocol.py`)

## Cross-model comparison

The rSLDS / TrSLDS / SHS-RSSM comparison on this dataset (shared Hamming /
NMI / ARI metrics, regime-change figures, tables) lives in `../compare/`:

```
cd ../compare
python run_shs.py    --dataset toyark13 --nseq 12
python run_trslds.py --dataset toyark13 --nseq 12
python run_rslds.py  --dataset toyark13 --nseq 12   # legacy env, see
                                                    # compare/environment-baselines.yml
python make_figures.py --dataset toyark13
```

Context for the numbers: ToyARK13 switches by a Markov chain with no spatial
structure -- the regime the HDP-HMM family targets -- so continuous-latent
recurrence (rSLDS/TrSLDS hyperplanes) is not expected to help here, and K is
fixed for those baselines while SHS adapts it.
