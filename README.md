# Practical Adversarial Training with Differential Privacy for Deep Learning

This is the code to train deep learning models with both differential privacy and adversarial robustness. We propose a adversarial (Adv) training procedure that incorporates differential privacy (DP), i.e. DP-Adv training.

The idea is simple: consider the minimax problem of adversarial training

<img src="https://render.githubusercontent.com/render/math?math=\min_\theta \max_\Delta L(f(x%2B\Delta,\theta),y)">
where L is loss, f is model, y is label, x is example, Δ is perturbation.

We apply traditional attacking method (FGSM, PGD, AutoAttack) for the inner maximization and DP optimizers (DP-SGD, DP-Adam) for the outer minimization.

The attacking method can be imported from [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).
```python
import torchattacks
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
adv_images = atk(images, labels)
```
The DP optimizers can be imported from Facebook library [Opacus](https://github.com/pytorch/opacus) or Google library [Tensorflow Privacy](https://github.com/tensorflow/privacy).

E.g. for Opacus,
```python
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
privacy_engine = PrivacyEngine(
    model,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1.3,
    max_grad_norm=1.0,
)
privacy_engine.attach(optimizer)
# Now it's business as usual
```

A tutorial on MNIST is available in [Tutorial: MNIST DP-Adv]([Github_Tutorial]DP+Adversarial_MNIST.ipynb)

We provide a pseudo-code and claim in paper that DP-Adv is as private as regular DP training, and as fast as regular adversarial training in clock time.

![DP-Adv=FGSM+DP-SGD](DP-Adv=FGSM+DPSGD.png?raw=true)

We also compare with concurrent work [StoBatch](https://github.com/haiphanNJIT/StoBatch) in "Scalable Differential Privacy with Certified Robustness in Adversarial Learning" and study the effects of Adv and/or DP on the calibration of networks.
