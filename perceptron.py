from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Perceptron(Chain):
    def __init__(self, input_unit, n_unit, output_unit):
        super(Perceptron, self).__init__(
            l1 = L.Linear(input_unit, n_unit),
            l2 = L.Linear(n_unit, n_unit),
            l3 = L.Linear(n_unit,output_unit)
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        o = self.l3(h2)
        return o
