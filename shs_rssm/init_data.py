import torch


@torch.no_grad()
def init_from_random_blocks(head, corpus, K, seed=0, block_len=None):
    g = torch.Generator().manual_seed(int(seed))
    head.stat_store.reset()
    for _e in corpus:
        bid, z, h, isf = _e[0], _e[1], _e[2], _e[3]
        _v = _e[5] if len(_e) > 5 else None
        B, T, _ = z.shape
        bl = int(block_len or max(10, T // (2 * K)))
        lab = torch.zeros(B, T, dtype=torch.long)
        for r in range(B):
            pos, order, i = 0, torch.randperm(K, generator=g), 0
            while pos < T:
                n = int(torch.randint(max(1, bl // 2), 2 * bl, (1,),
                                      generator=g).item())
                lab[r, pos:pos + n] = order[i % K]
                pos += n
                i += 1
        lab = lab.to(z.device)
        gam = torch.zeros(B, T, K, dtype=z.dtype, device=z.device)
        gam.scatter_(2, lab.unsqueeze(-1), 1.0)
        isf_bt = None
        if isf is not None:
            isf_bt = isf.reshape(B, T).to(device=z.device) > 0.5
        val_bt = None
        if _v is not None:
            val_bt = _v.reshape(B, T).to(device=z.device) > 0.5
            gam = gam * val_bt.unsqueeze(-1).to(gam.dtype)
        cnt = torch.zeros(K, K, dtype=torch.float64, device=z.device)
        if T > 1:
            link = torch.ones(B, T - 1, dtype=torch.bool, device=z.device)
            if isf_bt is not None:
                link &= ~isf_bt[:, 1:]
            if val_bt is not None:
                link &= val_bt[:, 1:] & val_bt[:, :-1]
            a = lab[:, :-1][link]
            b = lab[:, 1:][link]
            if a.numel():
                cnt.view(-1).scatter_add_(
                    0, a * K + b, torch.ones(a.numel(), dtype=torch.float64,
                                             device=z.device))
        starts = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        starts[:, 0] = True
        if isf_bt is not None:
            starts |= isf_bt
        if val_bt is not None:
            starts &= val_bt
        sc = torch.zeros(K, dtype=torch.float64, device=z.device)
        s_lab = lab[starts]
        if s_lab.numel():
            sc.scatter_add_(0, s_lab, torch.ones(s_lab.numel(),
                                                 dtype=torch.float64,
                                                 device=z.device))
        head.update_globals(z, h, gam, cnt, sc, is_first=isf, valid=_v,
                            batch_id=bid, stats_only=True)
    head.global_step_from_totals()
    return head

@torch.no_grad()
def init_contig_blocks(head, corpus, K, seed=0, block_len=20):
    g = torch.Generator().manual_seed(int(seed))
    rows = []
    for ci, e in enumerate(corpus):
        z = e[1]; v = e[5] if len(e) > 5 else None
        for r in range(z.shape[0]):
            T = int(z.shape[1]) if v is None else int(v[r].sum().item())
            if T > 1:
                rows.append((ci, r, T))
    if not rows:
        raise ValueError("no usable sequences in corpus")
    order = torch.randperm(len(rows), generator=g).tolist()
    wins = {}
    for k in range(K):
        ci, r, T = rows[order[k % len(rows)]]
        if block_len >= T:
            a, b = 0, T
        else:
            a = int(torch.randint(0, T - block_len, (1,), generator=g).item())
            b = a + block_len
        wins.setdefault(ci, []).append((k, r, a, b))
    head.stat_store.reset()
    mass = torch.zeros(K, dtype=torch.float64)
    for ci, e in enumerate(corpus):
        bid, z, h, isf = e[0], e[1], e[2], e[3]
        vm = e[5] if len(e) > 5 else None
        gam = torch.zeros(z.shape[0], z.shape[1], K, dtype=z.dtype, device=z.device)
        for (k, r, a, b) in wins.get(ci, []):
            gam[r, a:b, k] = 1.0
            mass[k] += float(b - a)
        nrow = max(z.shape[0], 1)
        cnt = torch.ones(K, K, dtype=torch.float64) / (len(corpus) * nrow)
        sc = torch.ones(K, dtype=torch.float64) / (len(corpus) * nrow)
        head.update_globals(z, h, gam, cnt, sc, is_first=isf, valid=vm,
                            batch_id=bid, stats_only=True)
    head.global_step_from_totals()
    if int((mass == 0).sum()):
        raise ValueError(f"{int((mass==0).sum())} of {K} regimes got no data; lower block_len or K")
    return head, dict(assigned_mass=mass)
