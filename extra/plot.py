import numpy as np
import scipy.special
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

activation_functions = {
    'sigmoid': {
        'func': lambda z: 1 / (1 + np.exp(-z)),
        'title': 'Sigmoid Activation Function',
        'label': '$\\sigma(z) = \\frac{1}{1+e^{-z}}$'
    },
    'relu': {
        'func': lambda z: np.maximum(0, z),
        'title': 'Rectified Linear Unit Activation Function (ReLU)',
        'label': '$ReLU(z) = max(0, z)$'
    },
    'gelu': {
        'func': lambda x: 0.5 * x * (1 + scipy.special.erf(x / np.sqrt(2))),
        'title': 'Gaussian Error Linear Unit Activation Function (GELU)',
        'label': r'$\text{GELU}(z) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right)$'}
}


def plot_activation_functions():
    z = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 9))
    for i, (_, value) in enumerate(activation_functions.items(), 1):
        y = value['func'](z)
        plt.plot(z, y, label=value['label'], linewidth=5)
        plt.xlabel('$z$')
        plt.ylabel('$y$')

    plt.legend()
    plt.grid(True)
    plt.title('Activation Functions')
    plt.tight_layout()
    plt.savefig('activation_functions.png', bbox_inches='tight', dpi=300)
    plt.show()


def main():
    plot_activation_functions()


if __name__ == '__main__':
    main()
