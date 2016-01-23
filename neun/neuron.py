import math

from exceptions import InvalidNeuronException

class Neuron(object):
    """
     Empty Neuron().fire() = 0 by default
     To get an active Neuron use : Neuron(value=1)
     weights are not counted unless Neuron is connected to another Neuron.
    """
    #@TODO: May be separate models into separate object that can take extra params, i.e in segmoid : precision
    #@TODO: by default neuron should be linear and return its value, only the final tail is s et to one of the intended models
    #@TODO: when bias is enabled, insert to inputs one neuron with value=1 and weight=bias
    #@todo: may be start with shreshold = 0
    LINEAR = 'LINEAR'
    BINARY_THRESHOLD = 'BINARY_THRESHOLD'
    RECTIFIED_LINEAR = 'RECTIFIED_LINEAR'
    SEGMOID = 'SEGMOID'
    STOCHASTIC = 'STOCHASTIC'
    
    def __init__(self, value=0, threshold=1, weight=0, bias=0, model=BINARY_THRESHOLD, neuron_input_list=[]):
        self.threshold = threshold
        self.weight = weight
        self.set_inputs(neuron_input_list)
        self.model = model
        self.bias = bias
        self.value = value

    def set_inputs(self, neuron_input_list):
        self.inputs = neuron_input_list
    
    def fire(self):
        total = self.value
        for neuron in self.inputs:
            if not isinstance(neuron, Neuron):
                raise InvalidNeuronException()
            total += neuron.fire() * neuron.weight
        return Neuron.get_value(self, total)
    
    @staticmethod
    def and_gate():
        #@TODO: implement a mecnanism to raise threshold according to number of inputs
        return Neuron(threshold=1.5)
    
    @staticmethod
    def or_gate():
        return Neuron(threshold=0.9)

    @staticmethod
    def xor_gate():
        in1 = Neuron(weight=1, threshold=0.5)
        in2 = Neuron(weight=-1, threshold=1.5)
        xor = Neuron(threshold=0.5, neuron_input_list=[in1, in2])
        
        def set_inputs(self):
            def func(neuron_input_list):
                self.inputs[0].set_inputs(neuron_input_list)
                self.inputs[1].set_inputs(neuron_input_list)
            return func
        
        xor.set_inputs = set_inputs(xor)
        return xor
    
    @staticmethod
    def _get_LINEAR_calculator(self, total):
        return total + self.bias
    
    @staticmethod
    def _get_BINARY_THRESHOLD_calculator(self, total):
        if self.bias:
            total = total + self.bias
            if total < 0:
                return 0
            return 1
        if total < self.threshold:
            return 0
        return 1

    @staticmethod
    def _get_RECTIFIED_LINEAR_calculator(self, total):
            total = total + self.bias
            if total > 0:
                return total
            return 0                

    @staticmethod
    def _get_SEGMOID_calculator(self, total):
        try:
            return round(1.0/(1.0 + math.exp(-1 * (total + self.bias))), 6)
        except OverflowError:
            return 0.0
    
    @staticmethod
    def _get_STOCHASTIC_calculator():
        pass

    @staticmethod
    def get_value(self, total):
        return getattr(Neuron, '_get_%s_calculator' % self.model)(self, total)
