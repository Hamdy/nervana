import unittest
from neun.neuron import Neuron

class BasicFunctionalities(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_binary_threshold_model(self):
        #- By Default we use Binary Threshold Model -#
        
        #- No Bias provided, Value >= Threshold -> 1 else 0 -#

        input1 = Neuron(value=1)
        input2 = Neuron(value=1)
        self.assertEqual(input1.fire(), 1)
        self.assertEqual(input2.fire(), 1)
        
        input1 = Neuron(value=0)
        input2 = Neuron(value=0)
        self.assertEqual(input1.fire(), 0)
        self.assertEqual(input2.fire(), 0)
        
        input1 = Neuron()
        input2 = Neuron()
        self.assertEqual(input1.fire(), 0)
        self.assertEqual(input2.fire(), 0)
        
        
        input1 = Neuron(value=2)
        input2 = Neuron(value=2)
        self.assertEqual(input1.fire(), 1)
        self.assertEqual(input2.fire(), 1)
        
        input1 = Neuron(value=-1)
        input2 = Neuron(value=-1)
        self.assertEqual(input1.fire(), 0)
        self.assertEqual(input2.fire(), 0)
        
        #- Bias provided, Value >= 0 -> 1 else 0 -#
        
        input1 = Neuron(value=1, bias=1)
        input2 = Neuron(value=1, bias=1)
        self.assertEqual(input1.fire(), 1)
        self.assertEqual(input2.fire(), 1)
        
        input1 = Neuron(value=0, bias=1)
        input2 = Neuron(value=0, bias=1)
        self.assertEqual(input1.fire(), 1)
        self.assertEqual(input2.fire(), 1)

    
    def test_linear_model(self):
        n1 = Neuron(model=Neuron.LINEAR)
        self.assertEquals(n1.fire(), 0)
        
        n2 = Neuron(model=Neuron.LINEAR, value=0)
        self.assertEquals(n2.fire(), 0)
        
        n3 = Neuron(model=Neuron.LINEAR, value=1)
        self.assertEquals(n3.fire(), 1)
        
        n4 = Neuron(model=Neuron.LINEAR, value=3)
        self.assertEquals(n4.fire(), 3)
        
        n5 = Neuron(model=Neuron.LINEAR, value=-1)
        self.assertEquals(n5.fire(), -1)
        
        n6 = Neuron(model=Neuron.LINEAR, value=2, bias=3)
        self.assertEquals(n6.fire(), 5)
        
    def test_segmoid(self):
        n1 = Neuron(model=Neuron.SEGMOID)
        self.assertEquals(n1.fire(), 0.5)
        
        n2 = Neuron(value=200, model=Neuron.SEGMOID)
        self.assertEquals(n2.fire(), 1)
        
        n3 = Neuron(value=2000, model=Neuron.SEGMOID)
        self.assertEquals(n3.fire(), 1)
        
        n4 = Neuron(value=20000000000, model=Neuron.SEGMOID)
        self.assertEquals(n4.fire(), 1)
        
        n5 = Neuron(value=-200, model=Neuron.SEGMOID)
        self.assertEquals(n5.fire(), 0)
        
        n6 = Neuron(value=-2000, model=Neuron.SEGMOID)
        self.assertEquals(n6.fire(), 0)
        
        n7 = Neuron(value=-2000000, model=Neuron.SEGMOID)
        self.assertEquals(n7.fire(), 0)
        
        n8 = Neuron(value=-2000000, bias = 10, model=Neuron.SEGMOID)
        self.assertEquals(n8.fire(), 0)
        
        n9= Neuron(value=2000000, bias = 10, model=Neuron.SEGMOID)
        self.assertEquals(n9.fire(), 1)
        
        n10 = Neuron(value=5, bias=-5, model=Neuron.SEGMOID)
        self.assertEquals(n10.fire(), 0.5)
        
        n11 = Neuron(value=-200, bias=200, model=Neuron.SEGMOID)
        self.assertEquals(n11.fire(), 0.5)

    def test_rectified_linear_model(self):
        n1 = Neuron(model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n1.fire(), 0)
        
        n2 = Neuron(value=1, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n2.fire(), 1)
        
        n3 = Neuron(value=2, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n3.fire(), 2)

        n4 = Neuron(value=10, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n4.fire(), 10)
                
        n5 = Neuron(value=-1, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n5.fire(), 0)
        
        n6 = Neuron(value=-10, bias=10, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n6.fire(), 0)
        
        n7 = Neuron(bias=10, value=-10, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n7.fire(), 0)
        
        n8 = Neuron(value=20, bias=-19, model=Neuron.RECTIFIED_LINEAR)
        self.assertEquals(n8.fire(), 1)

        
    def test_and_gate(self):
                        
        input3 = Neuron(value=1, weight=1) #1
        input4 = Neuron(value=1, weight=1) #1
        self.assertEqual(input3.fire(), 1)
        self.assertEqual(input4.fire(), 1)
        
        and_gate = Neuron(threshold=1.5, neuron_input_list=[input3, input4])
        self.assertEqual(and_gate.fire(), 1) # True & True = True
        
        input5 = Neuron(value=0, weight=1) #0
        input6 = Neuron(value=1, weight=1) #1
        self.assertEqual(input5.fire(), 0)
        self.assertEqual(input6.fire(), 1)
        and_gate2 = Neuron(threshold=1.5, neuron_input_list=[input5, input6])
        self.assertEqual(and_gate2.fire(), 0) # True & False = False
        
        and_gate3 = Neuron(threshold=1.5, neuron_input_list=[input6, input5])
        self.assertEqual(and_gate3.fire(), 0) # False & True = False
        
        input7 = Neuron(weight=1) #0
        input8 = Neuron(weight=1) #0
        self.assertEqual(input7.fire(), 0)
        self.assertEqual(input8.fire(), 0)
        
        and_gate4 = Neuron(threshold=1.5, neuron_input_list=[input7, input8])
        self.assertEqual(and_gate4.fire(), 0) # False & False = False
        
        #- Test and API -#
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(value=1, weight=1) #1
        and_gate = Neuron.and_gate()
        and_gate.set_inputs([in1, in2])
        self.assertEqual(and_gate.fire(), 1)
        
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(weight=1) #0
        self.assertEquals(in1.fire(), 1)
        self.assertEquals(in2.fire(), 0)
        
        and_gate = Neuron.and_gate()
        and_gate.set_inputs([in1, in2])
        self.assertEqual(and_gate.fire(), 0)
        
        and_gate.set_inputs([in2, in1])
        self.assertEqual(and_gate.fire(), 0)
        
        and_gate.set_inputs([in2, in2])
        self.assertEqual(and_gate.fire(), 0)

    def test_or_gate(self):
        input1 = Neuron(value=1, weight=1)
        input2 = Neuron(value=1, weight=1)
        or_gate = Neuron(threshold=.09, neuron_input_list=[input1, input2])
        self.assertEqual(or_gate.fire(), 1) # True or True = True
        
        input3 = Neuron(weight=1) # 0
        input4 = Neuron(value=1, weight=1) # 1
        or_gate2 = Neuron(threshold=.09, neuron_input_list=[input3, input4])
        self.assertEqual(or_gate2.fire(), 1) # False or True = True
        
        or_gate3 = Neuron(threshold=.09, neuron_input_list=[input4, input3])
        self.assertEqual(or_gate3.fire(), 1) # True or False = True

        input5 = Neuron(weight=1) # 0
        input6 = Neuron(weight=1) # 0
        or_gate4 = Neuron(threshold=.09, neuron_input_list=[input5, input6])
        self.assertEqual(or_gate4.fire(), 0) # False or False = False
        
        #- OR API -#
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(value=1, weight=1) #1
        or_gate = Neuron.or_gate()
        or_gate.set_inputs([in1, in2])
        self.assertEqual(or_gate.fire(), 1)
        
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(weight=1) #0
        or_gate = Neuron.or_gate()
        or_gate.set_inputs([in1, in2])
        self.assertEqual(or_gate.fire(), 1)
        
        or_gate.set_inputs([in2, in1])
        self.assertEqual(or_gate.fire(), 1)
        
        or_gate.set_inputs([in2, in2])
        self.assertEqual(or_gate.fire(), 0)

    def test_xor_gate(self):
        input1 = Neuron(value=1, weight=1) #1
        input2 = Neuron(value=1, weight=1) #1
        input3 = Neuron(weight=1, threshold=0.5, neuron_input_list=[input1, input2])
        input4 = Neuron(weight=-1, threshold=1.5, neuron_input_list=[input1, input2])
        xor_gate = Neuron(threshold=0.5, neuron_input_list=[input3, input4])
        self.assertEquals(xor_gate.fire(), 0)
        
        
        input1 = Neuron(value=1, weight=1) #1
        input2 = Neuron(weight=1) #0
        input3 = Neuron(weight=1, threshold=0.5, neuron_input_list=[input1, input2])
        input4 = Neuron(weight=-1, threshold=1.5, neuron_input_list=[input1, input2])
        xor_gate = Neuron(threshold=0.5, neuron_input_list=[input3, input4])
        self.assertEquals(xor_gate.fire(), 1)
        
        input1 = Neuron(value=1, weight=1) #1
        input2 = Neuron(weight=1) #0
        input3 = Neuron(weight=1, threshold=0.5, neuron_input_list=[input2, input1])
        input4 = Neuron(weight=-1, threshold=1.5, neuron_input_list=[input1, input2])
        xor_gate = Neuron(threshold=0.5, neuron_input_list=[input4, input3])
        self.assertEquals(xor_gate.fire(), 1)
        
        input1 = Neuron(weight=1) #0
        input2 = Neuron(weight=1) #0
        input3 = Neuron(weight=1, threshold=0.5, neuron_input_list=[input1, input2])
        input4 = Neuron(weight=-1, threshold=1.5, neuron_input_list=[input1, input2])
        xor_gate = Neuron(threshold=0.5, neuron_input_list=[input3, input4])
        self.assertEquals(xor_gate.fire(), 0)
        
        #- XOR API -#
        
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(value=1, weight=1) #1
        xor_gate = Neuron.xor_gate()
        xor_gate.set_inputs([in1, in2])
        self.assertEqual(xor_gate.fire(), 0)
        
        in1 = Neuron(value=1, weight=1) #1
        in2 = Neuron(weight=1) #0
        xor_gate = Neuron.xor_gate()
        xor_gate.set_inputs([in1, in2])
        self.assertEqual(xor_gate.fire(), 1)
        
        xor_gate.set_inputs([in2, in1])
        self.assertEqual(xor_gate.fire(), 1)
        
        xor_gate.set_inputs([in2, in2])
        self.assertEqual(xor_gate.fire(), 0)
        
    def test_excercise_coursera(self):
        """
        x1 = 2, w1 = 1
        x2 = 1, w2 = 0.5
        x3 = 1, w3 = 0
        bias = 0.5
        
        sum(xi*wi) + bias = 3
        
        Linear Model => output=3
        Binary Threshold Model => output=1
        Rectified Linear Model => output=3
        Segmoid Model => output= 0.95 
        """
        
        #- Linear -#
        n1 = Neuron(value=2, weight=1, model=Neuron.LINEAR)
        n2 = Neuron(value=1, weight=0.5, model=Neuron.LINEAR)
        n3 = Neuron(value=1, weight=0, model=Neuron.LINEAR)
        
        linear_model = Neuron(bias=0.5, model=Neuron.LINEAR, neuron_input_list=[n1, n2, n3])
        self.assertEquals(linear_model.fire(), 3)
        
        #- Binary Threshold -#
        n1 = Neuron(value=2, weight=1, model=Neuron.LINEAR)
        n2 = Neuron(value=1, weight=0.5, model=Neuron.LINEAR)
        n3 = Neuron(value=1, weight=0, model=Neuron.LINEAR)
        
        linear_model = Neuron(bias=0.5, model=Neuron.BINARY_THRESHOLD, neuron_input_list=[n1, n2, n3])
        self.assertEquals(linear_model.fire(), 1)
        
        
        #- Rectified  Linear -#
        
        n1 = Neuron(value=2, weight=1, model=Neuron.LINEAR)
        n2 = Neuron(value=1, weight=0.5, model=Neuron.LINEAR)
        n3 = Neuron(value=1, weight=0, model=Neuron.LINEAR)
        
        linear_model = Neuron(bias=0.5, model=Neuron.RECTIFIED_LINEAR, neuron_input_list=[n1, n2, n3])
        self.assertEquals(linear_model.fire(), 3)
        
        #- Segmoid -#
        
        n1 = Neuron(value=2, weight=1, model=Neuron.LINEAR)
        n2 = Neuron(value=1, weight=0.5, model=Neuron.LINEAR)
        n3 = Neuron(value=1, weight=0, model=Neuron.LINEAR)
        
        linear_model = Neuron(bias=0.5, model=Neuron.SEGMOID, neuron_input_list=[n1, n2, n3])
        self.assertEquals(round(linear_model.fire(), 2), .95)

if __name__ == "__main__":
    unittest.main()