
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)

def make_pair(D, angle_deg):
    a = normalize(rng.normal(size=D))
    u = rng.normal(size=D)
    u = u - np.dot(u, a) * a
    u = normalize(u)
    theta = np.deg2rad(angle_deg)
    b = np.cos(theta) * a + np.sin(theta) * u
    return a, b

def angle(a, b):
    c = np.dot(a, b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def q_binary(x):
    return normalize(np.where(x>=0,1,-1))

def q_ternary(x):
    s = np.std(x)+1e-12
    q = np.where(x>0.5*s,1,np.where(x<-0.5*s,-1,0))
    if np.linalg.norm(q)==0:
        q[np.argmax(np.abs(x))]=np.sign(x[np.argmax(np.abs(x))])
    return normalize(q)

def q_int3(x):
    m = np.max(np.abs(x))+1e-12
    q = np.clip(np.round(x/m*3),-4,3)
    return normalize(q)

quantizers = {"binary":q_binary,"ternary":q_ternary,"int3":q_int3}

dims_list = [256,512,1024,2048,4096,8192]
angles = np.linspace(1,179,18)
trials = 200

# plot mean distortion for each dimension (separate plots per quantizer)
for fmt, qfn in quantizers.items():
    plt.figure(figsize=(8,5))
    for D in dims_list:
        means = []
        for ang in angles:
            errs = []
            for _ in range(trials):
                a,b = make_pair(D, ang)
                qa, qb = qfn(a), qfn(b)
                errs.append(abs(angle(qa,qb)-ang))
            means.append(np.mean(errs))
        plt.plot(angles, means, marker='o', label=f"D={D}")
    
    plt.xlabel("Initial angle (deg)")
    plt.ylabel("Mean absolute distortion (deg)")
    plt.title(f"{fmt} distortion vs angle across dimensions")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
