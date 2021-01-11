# %%
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import matplotlib.pyplot as plt

# %%


class PredictiveCodingCell(nn.Module):
    """Organic Normalizaation w/ predictive coding
    dy/dt = -y + (b/(1+b))z + (1/(1+a))yhat + (a0/(1+a0))(1/N)[y - sum(yhat)]
    Turns out this model is no bueno because there is a hot potato winner-take all of activity going on between the
    two principal neurons.
    """

    def __init__(self, requires_grad=False):
        super().__init__()

        self.tau_y = Parameter(torch.rand(1) * 10., requires_grad=requires_grad)
        
        self.a0 = Parameter(torch.rand(1), requires_grad=requires_grad)
        self.b0 = Parameter(torch.rand(1), requires_grad=requires_grad)

        self.a_min = 0
        self.b_min = 0

        self.omega = Parameter(torch.rand(2), requires_grad=requires_grad)
        # self.Wyy = Parameter(torch.rand(6,6), requires_grad=requires_grad)

        # this will be set to true once parent inits weights
        self.weights_initialized = False

    def forward(self, z, hidden, delta_t=0.1):
        assert self.weights_initialized
        y, b, a, y_sum = hidden

        batch_size, n = y.shape

        # compute yhat
        Wyy = torch.eye(n) * (1. + 2j*np.pi*self.tau_y*self.omega)
        y_hat = torch.matmul(Wyy, y.T).T

        # delta_y_hat_sum = (delta_t/self.tau_y_hat) * (-y_hat_sum + y_hat.sum(dim=1, keepdim=True))
        # y_hat_sum_new = y_hat_sum + delta_y_hat_sum

        # compute predictive term y_real_sum
        y_sum = y.sum(dim=1, keepdim=True)

        # compute y
        delta_y = ((delta_t/self.tau_y) *
                   (-y + (b/(1.+b))*z
                       + (1./(1.+a))*y_hat
                       + (b/(1.+b)) * (y - y_sum.real))
                   )
        y_new = y + delta_y

        a_new = a
        b_new = b

        hidden = (y_new, b_new, a_new, y_sum)
        return hidden

# %%


class PredictiveCodingModel(nn.Module):

    def __init__(self, n, delta_t, requires_grad=False):
        super().__init__()

        self.delta_t = delta_t
        # recurrent step gets passed through
        self.filt = PredictiveCodingCell(requires_grad)
        self.n = n
        self.omega = torch.Tensor([2., 8.]) / (1000.)

    def forward(self, X, h0):
        # X: (batch_size, T, xi, theta, phi)
        # h0: (y, u, a, y_hat_sum) initial hidden state for each layer in cascade

        batch_size, T, n = X.shape
        outputs = [h0]
        # outputs = [tuple(map(torch.clone,h0))]

        def step(t):
            x = X[:, t]
            hidden = outputs[t - 1]

            if t == 30000:
                y, _, _, y_sum = hidden
                a = torch.zeros(1)
                b = torch.zeros(1)
                new_hidden = (y, b, a, y_sum)

                # shouldEqualOne = (y.real[0, 0]**2 + y.imag[0, 0]**2)**0.5
                # print(shouldEqualOne)
            else:
                new_hidden = hidden

            hidden = self.filt(x, new_hidden, self.delta_t)
            outputs.append(hidden)

        list(map(step, list(range(1, T))))
        y, b, a, y_sum = map(lambda x: torch.stack(x, dim=1), tuple(zip(*outputs)))

        norm1 = (y.real[0,30000,0]**2 + y.imag[0,30000,0]**2)**0.5
        y = y / norm1
        y_sum = y_sum / norm1

        return y, b, a, y_sum

    def init_hidden(self, batch_size, device=torch.device("cpu")):
        """Initialize the hidden state of RNN"""
        assert self.filt.weights_initialized, "Initialize weights before running"
        a_min = self.filt.a_min
        b_min = self.filt.b_min

        y0 = torch.zeros((batch_size, self.n), device=device).to(torch.complex64)
        # a0 = torch.ones((batch_size, self.n), device=device) * 0.01
        # b0 = torch.ones((batch_size, self.n), device=device) * 0.01
        a0 = torch.ones(1) * 0.01
        b0 = torch.ones(1) * 0.01

        y_sum = torch.zeros((batch_size, 1))

        return y0, b0, a0, y_sum  # this initializes `hidden`

    def init_weights(self):
        """Initialize the weights of the RNN cell"""
        # print('self.omega:', self.omega)
        state_dict = {'tau_y': torch.ones(1) * 5., # 10 msec
                      'b0': torch.ones(1)*0.01,
                      'a0': torch.ones(1)*0.01,
                      'omega': self.omega
                      }

        self.filt.load_state_dict(state_dict)
        self.filt.b_min = 0
        self.filt.a_min = 0
        self.filt.weights_initialized = True

# %%
def temporal_filter(x, n, tau, delta_t):
    """Cascade of exponential lowpass filters
    Parameters
    -----------
    x: torch.Tensor
        Input signal torch.Size([batch, time, h, w])
    n: int
        Number of filters in cascade
    tau: float
        time constant of exponential decay in filter

    Returns
    -------
    y: torch.Tensor
        Filtered output, with added dimension for number of filters in cascade
    """
    assert len(x.shape) == 4, 'Expected 4D tensor Size((b, T, h, w))'
    b, T, h, w = x.shape  # batch, time, height, width

    y_save = torch.zeros((b, T, h, w, n))
    y = torch.zeros((b, h, w, n))

    for t in range(T):
        delta_y = (delta_t/tau) * (-y[..., 0] + x[:, t])
        y[..., 0] = y[..., 0] + delta_y

        for nn in range(1, n):  # cascade
            delta_y = (delta_t/tau) * (-y[..., nn] + y[..., nn-1])
            y[..., nn] = y[..., nn] + delta_y

        y_save[:, t] = y
    return y_save

# %%
def input2(duration, dt, T, n):
    """Create input x
     
    """
    t1 = torch.arange(0, duration//2, dt)
    t2 = T - len(t1)

    # x1 has frequency 2Hz and x2 has frequency 8Hz
    x1 = np.cos(2./1000.*np.pi*2.*t1)
    x2 = -0.5*np.sin(2./1000.*np.pi*8.*t1)
    x = x1 + x2

    x_off = torch.zeros(t2)

    x = torch.cat([x, x_off], dim=0)

    # reshape to (T, n)
    x = x.view((T, 1)) * torch.ones((1, n))
    return x

def input(duration, dt, T, n):
    """ create input """
    t1 = torch.arange(0, duration, dt)

    x1 = np.cos(2./1000.*np.pi*2.*t1)
    x2 = -0.5*np.sin(2./1000.*np.pi*8.*t1)
    x = x1 + x2

    x = x.view((T, 1)) * torch.ones((1, n))

    return x

#%%

def noise_experiment():
    """Compare 100 noise trials vs clean trial"""
    seed = 0
    n = 2
    n_filt = 1
    tau_filt = 10.
    num_trials = 100

    duration = 6000.
    delta_t = 0.01
    contrast = 1
    t = torch.arange(0, duration, delta_t)
    T = len(t)

    # x1 = torch.zeros((T, n))
    # x1 = x1.view((1, T, n))

    x1 = input2(duration, delta_t, T, n)
    x1 = x1.view((1, T, n))

    mdl = PredictiveCodingModel(n=n, delta_t=delta_t)
    mdl.init_weights()
    h0 = mdl.init_hidden(batch_size=1)

    y_clean, _, _, y_sum = mdl(x1, h0)  # run the model
    y_real = y_clean.real
    y_imag = y_clean.imag

    fig, ax = plt.subplots(4, 1, sharex='all')
    # input
    ax[0].plot(t[::100], x1[0, ::100, 0], linewidth=.8)
    ax[0].set(ylabel='Input (x)')

    # real parts of y
    ax[1].plot(t[::100], y_real[0, ::100, 0], linewidth=.8)
    ax[1].plot(t[::100], y_imag[0, ::100, 0], linewidth=.8)

    # imaginary parts of y
    ax[2].plot(t[::100], y_real[0, ::100, 1], linewidth=.8)
    ax[2].plot(t[::100], y_imag[0, ::100, 1], linewidth=.8)

    # sum of responses
    ax[3].plot(t[::100], y_sum[0, ::100, 0].real, linewidth=.8)
    ax[3].plot(t[::100], y_sum[0, ::100, 0].imag, linewidth=.8)

    # ax[1].set(title='Real components')
    ax[1].set(ylabel='real y')
    ax[2].set(ylabel='imag y')
    ax[3].set(ylabel='Response (y)', xlabel='Time (ms)')
    fig.suptitle('Normalised predictive ORGaNICS')
    plt.show()


# %%
if __name__ == '__main__':

    noise_experiment()

# %%
