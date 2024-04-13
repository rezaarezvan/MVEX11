import numpy as np
import matplotlib.pyplot as plt

activation_functions = {
    'sigmoid': {
        'func': lambda x: 1 / (1 + np.exp(-x)),
        'title': 'Sigmoid',
        'label': '$\\sigma(x) = \\frac{1}{1+e^{-x}}$'
    },
    'tanh': {
        'func': np.tanh,
        'title': 'Tanh',
        'label': '$\\tanh(x)$'
    },
    'relu': {
        'func': lambda x: np.maximum(0, x),
        'title': 'ReLU',
        'label': '$ReLU(x) = max(0, x)$'
    },
    'gaussian': {
        'func': lambda x: np.exp(-x**2),
        'title': 'Gaussian',
        'label': '$e^{-x^2}$'
    }
}


def plot_activation_functions():
    x = np.linspace(-10, 10, 1000)
    plt.figure(figsize=(10, 8))
    plt.rcParams['text.usetex'] = True

    for i, (_, value) in enumerate(activation_functions.items(), 1):
        y = value['func'](x)
        plt.subplot(2, 2, i)
        plt.plot(x, y, label=value['label'])
        plt.title(value['title'])
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig('activation_functions.pdf')
    plt.show()


def main():
    plot_activation_functions()


if __name__ == '__main__':
    main()
