import numpy as np
from itertools import product
p=np.array([.1,.2,.3,.4]); C=np.array([True,False,True,True]); R=p[C]; Z=R.sum(); target=R/Z
u=np.array([-.7,.2,.6]); q=np.exp(u)/np.exp(u).sum(); a=np.log(q)-np.log(R)
kl=lambda q,p: np.sum(q*np.log(q/p))
assert np.allclose(kl(q,R),kl(q,target)-np.log(Z))
assert np.allclose(kl(target,R),-np.log(Z))
assert np.allclose(np.log(target)-np.log(R),-np.log(Z))
assert np.allclose(np.log(Z)+np.log(target)-np.log(R),0)
H=-np.sum(q*np.log(q)); assert np.allclose(-np.sum(q*np.log(R)),H+kl(q,R))
score=np.eye(3)-q[None,:]
gkl=(q*a)@score
h=1e-6
finite=[]
for j in range(3):
 v=np.eye(3)[j]*h
 softmax=lambda t:np.exp(t)/np.exp(t).sum()
 finite.append((kl(softmax(u+v),target)-kl(softmax(u-v),target))/(2*h))
assert np.allclose(gkl,finite,atol=1e-9)
# Expected sampled TB semi-gradient for arbitrary fixed offset.
z=.84
gtb=np.sum(2*q[:,None]*(z+a)[:,None]*score,axis=0)
assert np.allclose(gtb,2*gkl)
for K in (2,3,4):
 grad=np.zeros(3)
 for ids in product(range(3),repeat=K):
  ids=np.array(ids); gaps=a[ids]; mean=gaps.mean()
  prob=q[ids].prod()
  grad+=prob*(2*(gaps-mean)[:,None]*score[ids]).mean(axis=0)
 assert np.allclose(grad,2*(K-1)/K*gkl)
# Filtering only changes sequence log density by a shared additive constant.
raw=np.array([.08,.6,.2,.12]); accepted=raw[C]/raw[C].sum()
gaps_raw=np.log(raw[C])-np.log(R); gaps_acc=np.log(accepted)-np.log(R)
assert np.allclose(gaps_raw-gaps_raw.mean(),gaps_acc-gaps_acc.mean())
# Finite group normalizer is the optimum of its quadratic, not true log Z.
zstar=-a.mean()
assert abs(np.mean(2*(zstar+a)))<1e-12
assert not np.isclose(zstar,np.log(Z))
# New fixed checkpoint preserves normalization of the online target.
p2=np.array([.2,.15,.4,.25]);assert np.allclose((p2*C/(p2[C].sum())).sum(),1)
# Rank bound in the LoRA expression.
rng=np.random.default_rng(0);B=rng.normal(size=(8,2));A=rng.normal(size=(2,7));assert np.linalg.matrix_rank(B@A)<=2
print('PASS: SFT identity; KL projection/minimum; log balance; optimal offset; finite-difference KL gradient; expected TB and finite-group gradients; filtered-policy centering; online normalization; LoRA rank bound.')
