"""bnpy-style `randcontigblocks` initialisation for AIME's RegimeHead.

AIME's offline path has no data-driven initialiser. With identity_init=True every
regime starts with identical (A_k, Q_k), so the first E-step is exactly symmetric in k
and nothing can break the tie: responsibilities stay uniform, the MAP path collapses to
one state, and results are bit-identical across seeds. bnpy avoids this with
`initname='randcontigblocks'` (supplement A: "selects subwindows of data sequences at
random and uses these to create the global likelihood parameters").
"""
import torch


@torch.no_grad()
def init_from_random_blocks(head, corpus, K, seed=0, block_len=None):
    """One global M-step from hard, randomly-placed contiguous-block assignments."""
    g = torch.Generator().manual_seed(int(seed))
    head.stat_store.reset()
    for bid, z, h, isf in corpus:
        B, T, _ = z.shape
        bl = int(block_len or max(10, T // (2 * K)))
        # random contiguous blocks cycling through states in a random order
        lab = torch.zeros(B, T, dtype=torch.long)
        pos, order = 0, torch.randperm(K, generator=g)
        i = 0
        while pos < T:
            n = int(torch.randint(bl // 2, 2 * bl, (1,), generator=g).item())
            lab[0, pos:pos + n] = order[i % K]
            pos += n; i += 1
        gam = torch.zeros(B, T, K, dtype=z.dtype)
        gam.scatter_(2, lab.unsqueeze(-1), 1.0)
        # hard transition / start counts consistent with `lab`
        cnt = torch.zeros(K, K, dtype=torch.float64)
        idx_prev, idx_next = lab[0, :-1], lab[0, 1:]
        for a, b in zip(idx_prev.tolist(), idx_next.tolist()):
            cnt[a, b] += 1.0
        sc = torch.zeros(K, dtype=torch.float64); sc[lab[0, 0]] = 1.0
        head.update_globals(z, h, gam, cnt, sc, is_first=isf,
                            batch_id=bid, stats_only=True)
    head.global_step_from_totals()
    return head
