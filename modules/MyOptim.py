import torch
import torch.nn as nn
import torch.functional as F
import math
import numpy as np

class MyAdam:
    def __init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0) -> None:
        self.v = {}   # first order of moment
        self.s = {}    # second order of moment
        self.t = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        for key in params.keys():
            self.v[key] = torch.zeros_like(params[key])
            self.s[key] = torch.zeros_like(params[key])
            self.t[key] = 0
    
    # def step_old(self, phi, phi_grads, lr):
    #     for i, phi_grad in zip(phi.keys(), phi_grads):
    #         with torch.no_grad():
    #             self.v[i] = self.beta1*self.v[i] + (1-self.beta1)*phi_grad
    #             self.s[i] = self.beta2*self.s[i] + (1-self.beta2)*torch.square(phi_grad)
    #             v_bias_corr = self.v[i]/(1-self.beta1**self.t)
    #             s_bias_corr = self.s[i]/(1-self.beta2**self.t)
    #             tmp = v_bias_corr/(torch.sqrt(s_bias_corr)+self.eps)
    #             phi[i] = phi[i] - lr*tmp
    
    def step(self, phi, phi_grads, lr):
        for i, grad in zip(phi.keys(), phi_grads):
            with torch.no_grad():
                if self.weight_decay != 0:
                    grad = grad.add(phi[i], alpha=self.weight_decay)
                self.t[i] += 1
                self.v[i].mul_(self.beta1).add_(grad, alpha=1-self.beta1)
                # self.s[i].mul_(self.beta2).addcmul_(grad, grad.conj(), value=1-self.beta2)
                self.s[i] = self.beta2*self.s[i] + (1-self.beta2)*grad**2
                denom = (torch.sqrt(self.s[i])/math.sqrt(1-self.beta2**self.t[i]))+self.eps
                step_size = lr/(1-self.beta1**self.t[i])
                phi[i] = phi[i] - step_size*self.v[i]/denom

