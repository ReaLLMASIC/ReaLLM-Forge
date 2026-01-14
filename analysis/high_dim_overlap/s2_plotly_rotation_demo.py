# analysis/high_dim_overlap/s2_plotly_rotation_demo.py
import argparse, math, numpy as np, pandas as pd

# ---------- Sampling utilities ----------
def normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0] = 1.0; return X / n

def sample_uniform_s2(n, rng): return normalize_rows(rng.normal(size=(n,3)))

def sample_cauchy_projected(mu, gamma, n, rng):
    Y = mu.reshape(1,3) + rng.standard_cauchy(size=(n,3)) * gamma
    return normalize_rows(Y)

def sample_gaussian_projected(mu, sigma, n, rng):
    Y = mu.reshape(1,3) + rng.normal(size=(n,3)) * sigma
    return normalize_rows(Y)

def _orthonormal_basis_from_mu(mu):
    mu = mu / np.linalg.norm(mu)
    # pick a helper not colinear
    if abs(mu[2]) < 0.9: h = np.array([0.0, 0.0, 1.0])
    else: h = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(mu, h); e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    return mu, e1, e2

def sample_vmf(mu, kappa, n, rng):
    """von Mises–Fisher on S^2 with mean direction mu (|mu|=1) and concentration kappa.
    Uses inverse CDF for cos(theta): t ~ proportional to exp(kappa t) on [-1,1]."""
    mu = mu / np.linalg.norm(mu)
    if kappa <= 1e-8: return sample_uniform_s2(n, rng)
    u = rng.random(size=n)
    e2k = np.exp(2.0 * kappa)
    t = -1.0 + (np.log1p(u * (e2k - 1.0)) / kappa)  # cos(theta)
    t = np.clip(t, -1.0, 1.0)
    phi = rng.random(size=n) * (2.0*np.pi)
    sint = np.sqrt(np.clip(1.0 - t*t, 0.0, 1.0))
    mu_hat, e1, e2 = _orthonormal_basis_from_mu(mu)
    X = (t[:,None] * mu_hat[None,:] +
         (sint*np.cos(phi))[:,None] * e1[None,:] +
         (sint*np.sin(phi))[:,None] * e2[None,:])
    return normalize_rows(X)

# ---------- Kernels / metrics ----------
def Z_beta_s2(beta): return 1.0 if beta==0 else math.sinh(beta)/beta
def kde_density_at(X_ref, X_eval, beta, Zb): return np.mean(np.exp(beta*(X_eval@X_ref.T))/Zb, axis=1)

def angular_coverage_overlap(A,B,theta_rad):
    c=math.cos(theta_rad); AB=A@B.T
    return float(0.5*(np.mean((AB>=c).any(1))+np.mean((AB.T>=c).any(1)))), float(np.mean((AB>=c).any(1))), float(np.mean((AB.T>=c).any(1)))

def spherical_fibonacci_points(N):
    i=np.arange(N); phi=(np.pi*(3.0-np.sqrt(5.0)))*i; z=1.0-(2.0*(i+0.5)/N); r=np.sqrt(np.maximum(0.0,1.0-z*z))
    x=r*np.cos(phi); y=r*np.sin(phi)
    return normalize_rows(np.vstack([x,y,z]).T)

def healpix_bin_ids(X, nside=32):
    try:
        import healpy as hp
        theta,phi=hp.vec2ang(X.T); pix=hp.ang2pix(nside, theta, phi, nest=False); return pix.astype(int), f"HEALPix nside={nside}"
    except Exception:
        Npix=int(12*nside*nside); grid=spherical_fibonacci_points(Npix); idx=np.argmax(X@grid.T,axis=1)
        return idx.astype(int), f"Fibonacci fallback (≈HEALPix {Npix} cells)"

def occupancy_jaccard(idsA, idsB):
    sA=set(idsA.tolist()); sB=set(idsB.tolist()); inter=len(sA&sB); uni=len(sA|sB); return (inter/uni if uni>0 else 0.0), inter, len(sA), len(sB), uni

def hausdorff_directed_from_AB(AB):
    dA=np.arccos(np.clip(AB.max(1),-1,1)).max(); dB=np.arccos(np.clip(AB.max(0),-1,1)).max(); return float(dA), float(dB)

def kde_kl_directional(A,B,beta,Zb,eps=1e-12):
    nA=A.shape[0]; AA=A@A.T; KAA=np.exp(beta*AA)/Zb; self_k=math.exp(beta)/Zb; pA=(KAA.sum(1)-self_k)/max(nA-1,1)
    qB=np.mean(np.exp(beta*(A@B.T))/Zb, axis=1)
    return float(np.mean(np.log(np.maximum(pA,eps))-np.log(np.maximum(qB,eps))))

def chord_overlap_details(A, B, alpha=0.9):
    def wrap(a): return np.mod(a,2*np.pi)
    def min_arc(angles, alpha=0.9):
        n=angles.size; k=max(1,int(math.ceil(alpha*n))); arr=np.sort(wrap(angles)); arr2=np.concatenate([arr,arr+2*np.pi])
        spans=arr2[k-1:n+k-1]-arr; j=int(np.argmin(spans)); return float(arr[j]), float(arr2[j+k-1]), float(spans[j])
    def inter_len(a1,b1,a2,b2):
        def split(a,b): return [(a,b)] if b<=2*np.pi else [(a,2*np.pi),(0.0,b-2*np.pi)]
        inter=0.0
        for x1,y1 in split(a1,b1):
            for x2,y2 in split(a2,b2):
                lo,hi=max(x1,x2),min(y1,y2)
                if hi>lo: inter += (hi-lo)
        return float(inter)
    muA=A.mean(0); muB=B.mean(0)
    if np.linalg.norm(muA)<1e-12 or np.linalg.norm(muB)<1e-12: return (0.0,0.0,0.0,0.0,0.0)
    muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
    n=np.cross(muA,muB)
    if np.linalg.norm(n)<1e-9:
        tmp=np.array([0,0,1.0]); 
        if abs(muA@tmp)>0.9: tmp=np.array([1.0,0,0])
        n=np.cross(muA,tmp)
    n/=np.linalg.norm(n)
    e1=muA-(muA@n)*n
    if np.linalg.norm(e1)<1e-12: e1=muB-(muB@n)*n
    e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    angA=np.arctan2(A@e2,A@e1); angB=np.arctan2(B@e2,B@e1)
    a1,b1,w1=min_arc(angA,0.9); a2,b2,w2=min_arc(angB,0.9)
    inter=inter_len(a1,b1,a2,b2)
    union=w1+w2-inter if w1+w2-inter>1e-12 else 1e-12
    ovl=inter/union; mA=inter/max(w1,1e-12); mB=inter/max(w2,1e-12)
    return ovl,mA,mB,w1,w2

# ---------- Utility ----------
def dynamic_title(metric_mode, data_mode, gen_model, params):
    if data_mode == "pt":
        gen_txt = "user .pt data"
    else:
        if gen_model == "vmf":
            gen_txt = f"vMF around μx (κ₁={params['kappa1']:.1f}, κ₂={params['kappa2']:.1f})"
        elif gen_model == "gaussian":
            gen_txt = f"Gaussian around μx, projected (σ₁={params['sigma1']:.2f}, σ₂={params['sigma2']:.2f})"
        else:
            gen_txt = f"Cauchy around μx, projected (γ₁={params['gamma1']:.2f}, γ₂={params['gamma2']:.2f})"
    head = "Merge" if metric_mode=="merge" else "Overlap"
    return f"<b>{head} on S² — {gen_txt}, B rotated about Z</b>"

def parse_metrics(s):
    ALL = ["bc","mmd","ovl","mean_angle","mean_cos","hausdorff","fib_jaccard","hp_jaccard","chord_overlap",
           "kl","cov_dir","haus_dir","fib_recall","hp_recall","chord_merge","prox_cov","mean_nn"]
    if s is None or s.strip().lower()=="all": return ALL
    toks = [t.strip().lower() for t in s.split(",") if t.strip()]
    # validate
    ok = [t for t in toks if t in ALL]
    return ok if ok else ALL

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser(description="S^2 Plotly demo with overlap/merge modes + isotropic generators + metric selection")
    ap.add_argument("--mode", choices=["gen","pt"], default="gen")
    ap.add_argument("--pt", type=str, default=None)
    ap.add_argument("--nA", type=int, default=1000); ap.add_argument("--nB", type=int, default=1000)
    # generator model
    ap.add_argument("--gen-model", choices=["vmf","gaussian","cauchy"], default="vmf")
    # vMF parameters
    ap.add_argument("--kappa1", type=float, default=20.0); ap.add_argument("--kappa2", type=float, default=20.0)
    # Gaussian parameters
    ap.add_argument("--sigma1", type=float, default=0.25); ap.add_argument("--sigma2", type=float, default=0.25)
    # Cauchy parameters
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gamma1", type=float, default=0.7); ap.add_argument("--gamma2", type=float, default=0.7)
    # colors / point size / draw subset
    ap.add_argument("--color-A", type=str, default="#1f77b4"); ap.add_argument("--color-B", type=str, default="#ff7f0e")
    ap.add_argument("--marker-size", type=int, default=4); ap.add_argument("--plot-subset", type=int, default=900)
    # layout sizing
    ap.add_argument("--fig-height", type=int, default=1100); ap.add_argument("--scene-frac", type=float, default=0.45)
    # metrics params
    ap.add_argument("--beta", type=float, default=12.0); ap.add_argument("--theta-deg", type=float, default=60.0)
    ap.add_argument("--prox-deg", type=float, default=15.0)
    ap.add_argument("--fib-N", type=int, default=1200); ap.add_argument("--healpix-nside", type=int, default=32)
    ap.add_argument("--alpha-chord", type=float, default=0.9)
    ap.add_argument("--angle-step", type=int, default=10); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metric-mode", choices=["overlap","merge"], default="merge")
    ap.add_argument("--metrics", type=str, default="all", help="Comma-separated list of metrics to compute/display (or 'all').")
    ap.add_argument("--out-html", type=str, default="s2_merge_plotly.html"); ap.add_argument("--out-csv", type=str, default="s2_rotation_metrics.csv")
    args=ap.parse_args()

    rng=np.random.default_rng(args.seed)
    mu_x=np.array([1.0,0,0])
    selected = parse_metrics(args.metrics)

    # --- Load or generate ---
    if args.mode=="pt":
        import torch
        obj=torch.load(args.pt, map_location="cpu")
        if isinstance(obj,dict) and "A" in obj and "B" in obj: A=obj["A"].cpu().numpy(); B0=obj["B"].cpu().numpy()
        elif isinstance(obj,(list,tuple)) and len(obj)==2: A=obj[0].cpu().numpy(); B0=obj[1].cpu().numpy()
        else: raise ValueError("Unsupported .pt structure.")
        A=normalize_rows(A); B0=normalize_rows(B0)
        if A.shape[1]!=3 or B0.shape[1]!=3: raise ValueError("Data must be (n,3) for S^2 demo.")
    else:
        if args.gen_model == "vmf":
            A = sample_vmf(mu_x, args.kappa1, args.nA, rng); B0 = sample_vmf(mu_x, args.kappa2, args.nB, rng)
        elif args.gen_model == "gaussian":
            A = sample_gaussian_projected(mu_x, args.sigma1, args.nA, rng); B0 = sample_gaussian_projected(mu_x, args.sigma2, args.nB, rng)
        else: # cauchy
            g1 = args.gamma1 if args.gamma is None else (args.gamma1 if args.gamma1!=0.7 else args.gamma)
            g2 = args.gamma2 if args.gamma is None else (args.gamma2 if args.gamma2!=0.7 else args.gamma)
            if g1 is None: g1 = args.gamma1
            if g2 is None: g2 = args.gamma2
            A = sample_cauchy_projected(mu_x, g1, args.nA, rng); B0 = sample_cauchy_projected(mu_x, g2, args.nB, rng)

    # --- Precompute common things (only if needed) ---
    beta=args.beta; Zb=Z_beta_s2(beta); theta_rad=math.radians(args.theta_deg); prox_rad=math.radians(args.prox_deg)
    angles_deg=list(range(0,181,max(1,int(args.angle_step))))

    # For KDE-based metrics
    need_kde = any(m in selected for m in ["bc","mmd","kl"])
    if need_kde:
        X_eval=sample_uniform_s2(2000,rng); pA_eval=kde_density_at(A,X_eval,beta,Zb)
        AA=A@A.T; np.fill_diagonal(AA,-np.inf); kAA=np.sum(np.exp(beta*AA)[AA>-np.inf])/(A.shape[0]*(A.shape[0]-1))
        BB=B0@B0.T; np.fill_diagonal(BB,-np.inf); kBB=np.sum(np.exp(beta*BB)[BB>-np.inf])/(B0.shape[0]*(B0.shape[0]-1))

    # For binning
    need_bins = any(m in selected for m in ["fib_jaccard","fib_recall","hp_jaccard","hp_recall"])
    if need_bins:
        fib_grid=spherical_fibonacci_points(args.fib_N)
        fib_ids_A=np.argmax(A@fib_grid.T,axis=1)
        hp_ids_A, hp_label=healpix_bin_ids(A, args.healpix_nside)
    else:
        hp_label="HEALPix"

    # Containers for metrics to export
    export_dict = {"angle_deg": []}
    for ang in angles_deg: export_dict["angle_deg"].append(ang)

    # Storage for plotting
    store = {}

    # Compute per-angle metrics as needed
    for ang in angles_deg:
        R=np.array([[math.cos(math.radians(ang)),-math.sin(math.radians(ang)),0.0],
                    [math.sin(math.radians(ang)), math.cos(math.radians(ang)),0.0],
                    [0.0,0.0,1.0]])
        B=B0@R.T
        AB=A@B.T

        # means, symmetric stuff that many panels use
        if any(m in selected for m in ["mean_angle","mean_cos","hausdorff","ovl","cov_dir","prox_cov","mean_nn","haus_dir"]):
            muA=A.mean(0); muB=B.mean(0); muA/=np.linalg.norm(muA); muB/=np.linalg.norm(muB)
            cos_m=float(np.clip(muA@muB,-1,1))
            store.setdefault("mean_cos", []).append(cos_m)
            store.setdefault("mean_angle", []).append(math.degrees(math.acos(cos_m)))
        if "hausdorff" in selected or "haus_dir" in selected:
            hAB,hBA=hausdorff_directed_from_AB(AB)
            if "hausdorff" in selected: store.setdefault("haus_sym", []).append(math.degrees(max(hAB,hBA)))
            if "haus_dir" in selected:
                store.setdefault("haus_AtoB", []).append(math.degrees(hAB))
                store.setdefault("haus_BtoA", []).append(math.degrees(hBA))

        # overlap / coverage
        if "ovl" in selected or "cov_dir" in selected:
            cth=math.cos(theta_rad); covA=float(np.mean((AB>=cth).any(1))); covB=float(np.mean((AB.T>=cth).any(1)))
            if "ovl" in selected: store.setdefault("ovl", []).append(0.5*(covA+covB))
            if "cov_dir" in selected:
                store.setdefault("cov_A", []).append(covA); store.setdefault("cov_B", []).append(covB)

        # KDE BC / MMD / KL
        if "bc" in selected:
            qB_eval=kde_density_at(B, X_eval, beta, Zb); store.setdefault("bc", []).append(float(np.mean(np.sqrt(pA_eval*qB_eval))))
        if "mmd" in selected:
            kAB=np.mean(np.exp(beta*AB)); store.setdefault("mmd2", []).append(float((kAA+kBB-2*kAB)/Zb))
        if "kl" in selected:
            store.setdefault("kl_AtoB", []).append(kde_kl_directional(A,B,beta,Zb))
            store.setdefault("kl_BtoA", []).append(kde_kl_directional(B,A,beta,Zb))

        # binning
        if "fib_jaccard" in selected or "fib_recall" in selected:
            if "fib_grid" not in locals(): fib_grid=spherical_fibonacci_points(args.fib_N)
            fib_ids_B=np.argmax(B@fib_grid.T,axis=1); jacc, inter, nAocc, nBocc, uni=occupancy_jaccard(fib_ids_A,fib_ids_B)
            if "fib_jaccard" in selected: store.setdefault("fib_jaccard", []).append(jacc)
            if "fib_recall" in selected:
                store.setdefault("fib_recall_AinB", []).append(inter/max(nAocc,1)); store.setdefault("fib_recall_BinA", []).append(inter/max(nBocc,1))
        if "hp_jaccard" in selected or "hp_recall" in selected:
            hp_ids_B,_=healpix_bin_ids(B, args.healpix_nside); jh, ih, nAocc_h, nBocc_h, uh=occupancy_jaccard(hp_ids_A,hp_ids_B)
            if "hp_jaccard" in selected: store.setdefault("hp_jaccard", []).append(jh)
            if "hp_recall" in selected:
                store.setdefault("hp_recall_AinB", []).append(ih/max(nAocc_h,1)); store.setdefault("hp_recall_BinA", []).append(ih/max(nBocc_h,1))

        # chord
        if "chord_overlap" in selected or "chord_merge" in selected:
            ovl_sym,mA,mB,_,_=chord_overlap_details(A,B,args.alpha_chord)
            if "chord_overlap" in selected: store.setdefault("chord_ovl", []).append(ovl_sym)
            if "chord_merge" in selected:
                store.setdefault("chord_AtoB", []).append(mA); store.setdefault("chord_BtoA", []).append(mB)

        # proximity + mean nn
        if "prox_cov" in selected or "mean_nn" in selected:
            nearest_A=np.arccos(np.clip(AB.max(1),-1,1)); nearest_B=np.arccos(np.clip(AB.max(0),-1,1))
            if "prox_cov" in selected:
                store.setdefault("prox_AinB", []).append(float(np.mean(nearest_A <= prox_rad)))
                store.setdefault("prox_BinA", []).append(float(np.mean(nearest_B <= prox_rad)))
            if "mean_nn" in selected:
                store.setdefault("mean_nn_AtoB_deg", []).append(float(np.degrees(nearest_A.mean())))
                store.setdefault("mean_nn_BtoA_deg", []).append(float(np.degrees(nearest_B.mean())))

    # Build CSV export with only selected metrics
    export = {"angle_deg": angles_deg}
    if "bc" in selected: export["BC"] = store["bc"]
    if "mmd" in selected: export["MMD2_norm"] = store["mmd2"]
    if "ovl" in selected: export[f"OVL_theta_{args.theta_deg}deg"] = store["ovl"]
    if "mean_angle" in selected: export["mean_angle_deg"] = store["mean_angle"]
    if "mean_cos" in selected: export["mean_cosine"] = store["mean_cos"]
    if "hausdorff" in selected: export["Hausdorff_sym_deg"] = store["haus_sym"]
    if "fib_jaccard" in selected: export[f"Fib_Jaccard_N{args.fib_N}"] = store["fib_jaccard"]
    if "hp_jaccard" in selected: export[f"HP_Jaccard_{hp_label}"] = store["hp_jaccard"]
    if "chord_overlap" in selected: export[f"Chord_overlap_alpha{int(100*args.alpha_chord)}pct"] = store["chord_ovl"]
    if "kl" in selected: export["KL_AtoB"], export["KL_BtoA"] = store["kl_AtoB"], store["kl_BtoA"]
    if "cov_dir" in selected: export[f"Cov_A|B_theta_{args.theta_deg}deg"], export[f"Cov_B|A_theta_{args.theta_deg}deg"] = store["cov_A"], store["cov_B"]
    if "haus_dir" in selected: export["Haus_AtoB_deg"], export["Haus_BtoA_deg"] = store["haus_AtoB"], store["haus_BtoA"]
    if "fib_recall" in selected: export[f"Fib_recall_AinB_N{args.fib_N}"], export[f"Fib_recall_BinA_N{args.fib_N}"] = store["fib_recall_AinB"], store["fib_recall_BinA"]
    if "hp_recall" in selected: export[f"HP_recall_AinB_{hp_label}"], export[f"HP_recall_BinA_{hp_label}"] = store["hp_recall_AinB"], store["hp_recall_BinA"]
    if "chord_merge" in selected: export[f"Chord_merge_AtoB_alpha{int(100*args.alpha_chord)}pct"], export[f"Chord_merge_BtoA_alpha{int(100*args.alpha_chord)}pct"] = store["chord_AtoB"], store["chord_BtoA"]
    if "prox_cov" in selected: export[f"Prox{int(args.prox_deg)}_AinB"], export[f"Prox{int(args.prox_deg)}_BinA"] = store["prox_AinB"], store["prox_BinA"]
    if "mean_nn" in selected: export["MeanNN_AtoB_deg"], export["MeanNN_BtoA_deg"] = store["mean_nn_AtoB_deg"], store["mean_nn_BtoA_deg"]
    pd.DataFrame(export).to_csv(args.out_csv, index=False)

    # --- Plotly ---
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.offline import plot as plotly_save
    except Exception:
        print("Plotly not available; saved CSV only:", args.out_csv); return

    # dynamic grid: 1 row for scene + ceil(len(panels)/3) rows for metrics
    panels = []  # list of dicts: {title, y, y2, color_mode:'single'|'ab', y_label}
    neutral = ("#6C7A89", "#34495E", "#95A5A6", "#BDC3C7")

    def add_panel_single(key, title, y_vals, y_label, color_idx=0):
        panels.append({"title": title, "mode":"single", "ys":[y_vals], "y_label": y_label, "colors":[neutral[color_idx%len(neutral)]]})

    def add_panel_ab(key, title, yA, yB, y_label):
        panels.append({"title": title, "mode":"ab", "ys":[yA,yB], "y_label": y_label})

    # Populate panels according to selected metrics
    if "kl" in selected: add_panel_ab("kl","KDE-KL merge: KL(A||B) & KL(B||A)", store["kl_AtoB"], store["kl_BtoA"], "KL")
    if "cov_dir" in selected: add_panel_ab("cov","Angular coverage merges θ=%d°: Cov(A|B), Cov(B|A)"%int(args.theta_deg), store["cov_A"], store["cov_B"], "Coverage")
    if "haus_dir" in selected: add_panel_ab("hausd","Directed Hausdorff distances (deg)", store["haus_AtoB"], store["haus_BtoA"], "Hausdorff (deg)")
    if "fib_recall" in selected: add_panel_ab("fibr","Fibonacci bin recall: |A∩B|/|A| and /|B|", store["fib_recall_AinB"], store["fib_recall_BinA"], "Recall")
    if "hp_recall" in selected: add_panel_ab("hpr","HEALPix bin recall: |A∩B|/|A| and /|B|", store["hp_recall_AinB"], store["hp_recall_BinA"], "Recall")
    if "chord_merge" in selected: add_panel_ab("chm", f"Chord merges (α={int(100*args.alpha_chord)}%): inter/width_A and inter/width_B", store["chord_AtoB"], store["chord_BtoA"], "Merge (0–1)")

    if "bc" in selected: add_panel_single("bc","Bhattacharyya coefficient (KDE)", store["bc"], "BC (0–1)", 0)
    if "mmd" in selected: add_panel_single("mmd","Kernel MMD² (normalized)", store["mmd2"], "MMD²", 1)
    if "ovl" in selected: add_panel_single("ovl", f"Angular coverage overlap θ={int(args.theta_deg)}°", store["ovl"], "OVL (0–1)", 2)
    if "mean_angle" in selected: add_panel_single("ma","Angle between mean directions (deg)", store["mean_angle"], "Angle (deg)", 0)
    if "mean_cos" in selected: add_panel_single("mc","Cosine similarity of means", store["mean_cos"], "Cosine", 1)
    if "hausdorff" in selected: add_panel_single("hs","Hausdorff distance (deg, geodesic)", store["haus_sym"], "Hausdorff (deg)", 2)
    if "fib_jaccard" in selected: add_panel_single("fj", f"Fibonacci bins Jaccard (N={args.fib_N})", store["fib_jaccard"], "Jaccard", 0)
    if "hp_jaccard" in selected: add_panel_single("hj", f"{hp_label} Jaccard", store["hp_jaccard"], "Jaccard", 1)
    if "chord_overlap" in selected: add_panel_single("cho", f"Chord overlap on common great circle (α={int(100*args.alpha_chord)}%)", store["chord_ovl"], "Overlap (0–1)", 2)
    if "prox_cov" in selected: add_panel_ab("prox", f"Proximity coverage @ δ={int(args.prox_deg)}° (A|B & B|A)", store["prox_AinB"], store["prox_BinA"], "Prox. cov")
    if "mean_nn" in selected: add_panel_ab("mnn", "Mean nearest angle (deg)", store["mean_nn_AtoB_deg"], store["mean_nn_BtoA_deg"], "Mean NN angle (deg)")

    k = len(panels)
    metric_rows = (k + 2) // 3  # ceil(k/3)
    sf=max(0.05, min(0.9, float(args.scene_frac)))
    rest=(1.0-sf)/max(1,metric_rows)
    row_heights=[sf] + [rest]*metric_rows

    # Build titles list: 1 for scene + k + filler empties to reach 1 + metric_rows*3
    meta = {"kappa1":args.kappa1,"kappa2":args.kappa2,"sigma1":args.sigma1,"sigma2":args.sigma2,"gamma1":(args.gamma1 if args.gamma is not None else args.gamma1),"gamma2":(args.gamma2 if args.gamma is not None else args.gamma2)}
    title_html = dynamic_title(args.metric_mode, args.mode, args.gen_model, meta)
    subplot_titles = ["S² point clouds (A fixed, B rotated by φ about Z)"] + [p["title"] for p in panels]
    needed = metric_rows*3 - k
    subplot_titles += [""]*needed

    # Create figure
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    specs = [[{"type":"scene","colspan":3}, None, None]]
    for _ in range(metric_rows): specs.append([{"type":"xy"},{"type":"xy"},{"type":"xy"}])
    fig=make_subplots(rows=1+metric_rows, cols=3, row_heights=row_heights, specs=specs, subplot_titles=subplot_titles)
    fig.update_layout(title={"text":title_html,"x":0.5,"xanchor":"center"},
                      title_font_size=20, height=args.fig_height, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=60,b=80), showlegend=False)

    # 3D sphere and points
    u=np.linspace(0,2*np.pi,60); v=np.linspace(0,np.pi,30)
    uu,vv=np.meshgrid(u,v); xs=np.cos(uu)*np.sin(vv); ys=np.sin(uu)*np.sin(vv); zs=np.cos(vv)
    idxA=np.random.choice(A.shape[0], size=min(args.plot_subset, A.shape[0]), replace=False)
    idxB=np.random.choice(B0.shape[0], size=min(args.plot_subset, B0.shape[0]), replace=False)
    A_sub=A[idxA]; B0_sub=B0[idxB]; B_init=(B0_sub@np.eye(3).T)
    fig.add_trace(go.Surface(x=xs,y=ys,z=zs,opacity=0.2,showscale=False), row=1,col=1)
    fig.add_trace(go.Scatter3d(x=A_sub[:,0],y=A_sub[:,1],z=A_sub[:,2],mode="markers",
                               marker=dict(size=args.marker_size, color=args.color_A)), row=1,col=1)
    fig.add_trace(go.Scatter3d(x=B_init[:,0],y=B_init[:,1],z=B_init[:,2],mode="markers",
                               marker=dict(size=args.marker_size, color=args.color_B)), row=1,col=1)

    # Add metric traces panel by panel
    y_ranges=[]; v_coords=[]
    for i,p in enumerate(panels):
        r = 2 + (i//3); c = 1 + (i%3)
        if p["mode"]=="single":
            fig.add_trace(go.Scatter(x=angles_deg, y=p["ys"][0], mode="lines",
                                     line=dict(color=p["colors"][0])), row=r, col=c)
            ymin, ymax = float(np.min(p["ys"][0])), float(np.max(p["ys"][0]))
            if ymin==ymax: ymin, ymax = ymin-1e-6, ymax+1e-6
            y_ranges.append((ymin, ymax)); v_coords.append((r,c))
            fig.update_xaxes(title_text="φ (deg)", row=r, col=c); fig.update_yaxes(title_text=p["y_label"], row=r, col=c)
        else:  # ab (directional)
            # In merge mode use A/B colors, else neutral but two lines
            colA = args.color_A if args.metric_mode=="merge" else neutral[0]
            colB = args.color_B if args.metric_mode=="merge" else neutral[1]
            fig.add_trace(go.Scatter(x=angles_deg, y=p["ys"][0], mode="lines", line=dict(color=colA)), row=r, col=c)
            fig.add_trace(go.Scatter(x=angles_deg, y=p["ys"][1], mode="lines", line=dict(color=colB)), row=r, col=c)
            both = np.array(p["ys"][0])+np.array(p["ys"][1])
            ymin = float(min(np.min(p["ys"][0]), np.min(p["ys"][1])))
            ymax = float(max(np.max(p["ys"][0]), np.max(p["ys"][1])))
            if ymin==ymax: ymin, ymax = ymin-1e-6, ymax+1e-6
            y_ranges.append((ymin, ymax)); v_coords.append((r,c))
            fig.update_xaxes(title_text="φ (deg)", row=r, col=c); fig.update_yaxes(title_text=p["y_label"], row=r, col=c)

    # Vertical guides
    vline_indices=[]
    for (row,col),yr in zip(v_coords,y_ranges):
        fig.add_trace(go.Scatter(x=[angles_deg[0], angles_deg[0]], y=list(yr), mode="lines",
                                 line=dict(dash="dash", width=1), showlegend=False), row=row, col=col)
        vline_indices.append(len(fig.data)-1)

    # Frames
    frames=[]
    for ang in angles_deg:
        R=np.array([[math.cos(math.radians(ang)),-math.sin(math.radians(ang)),0.0],
                    [math.sin(math.radians(ang)), math.cos(math.radians(ang)),0.0],
                    [0.0,0.0,1.0]])
        B_sub=(B0_sub@R.T)
        updates=[go.Scatter3d(x=B_sub[:,0], y=B_sub[:,1], z=B_sub[:,2])]
        updates += [go.Scatter(x=[ang,ang], y=list(yr)) for yr in y_ranges]
        frames.append(go.Frame(name=str(ang), data=updates, traces=[2]+vline_indices))
    fig.frames=frames

    # Slider ("progress bar") and modebar config
    steps=[dict(method="animate", args=[[str(ang)],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}], label=f"{ang}°") for ang in angles_deg]
    sliders=[dict(active=0, currentvalue={"prefix":"Rotation φ = "}, pad={"t":30,"b":15}, steps=steps)]
    fig.update_layout(sliders=sliders)
    config=dict(displaylogo=False, modeBarButtonsToAdd=["autoScale2d","resetScale2d"], scrollZoom=True)

    from plotly.offline import plot as plotly_save
    plotly_save(fig, filename=args.out_html, auto_open=False, config=config)
    print("Wrote:", args.out_html); print("Wrote:", args.out_csv)

if __name__=="__main__":
    main()

