import numpy as np
import matplotlib.pyplot as plt

activation_functions = {
    'sigmoid': {
        'func': lambda z: 1 / (1 + np.exp(-z)),
        'title': 'Sigmoid Activation Function',
        'label': '$\\sigma(z) = \\frac{1}{1+e^{-z}}$'
    },
    'tanh': {
        'func': np.tanh,
        'title': 'Hyperbolic Tangent Activation Function',
        'label': '$\\tanh(z) = \\frac{e^z + e^{-z}}{e^z - e^{-z}}$'
    },
    'relu': {
        'func': lambda z: np.maximum(0, z),
        'title': 'Rectified Linear Unit Activation Function (ReLU)',
        'label': '$ReLU(z) = max(0, z)$'
    },
    'gaussian': {
        'func': lambda z: np.exp(-z**2),
        'title': 'Gaussian Activation Function',
        'label': '$f(z) = \\exp(-z^2)$'
    }
}


def plot_activation_functions():
    z = np.linspace(-10, 10, 1000)
    plt.figure(figsize=(10, 8))

    for i, (_, value) in enumerate(activation_functions.items(), 1):
        y = value['func'](z)
        plt.subplot(2, 2, i)
        plt.plot(z, y, label=value['label'])
        plt.title(value['title'])
        plt.xlabel('$z$')
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
