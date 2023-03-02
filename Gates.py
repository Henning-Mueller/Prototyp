class Component:


    def __init__(self, value=None):
        self.value = value
        self.outputs = []

    def calcValue(self):
        pass

    def getValue(self):
        return self.value

    def update(self):
        self.value= self.calcValue()
        if not self.outputs == None:
            for output in self.outputs:
                output.update()
    def appendOutput(self,n):
        self.outputs.append(n)

class Gate(Component):


    def __init__(self,inputs=[None, None]):
        self.inputs = [None, None]
        Component.__init__(self)
        for input in inputs:
            if not input == None:
                input.appendOutput(self)
                pass


    def addInput(self, input):
        if self.inputs[0] == None:
            self.inputs[0] = input
            self.inputs[0].appendOutput(self)
        elif self.inputs[1] == None:
            self.inputs[1] = input
            #todo make sure outputs exists
            self.inputs[1].appendOutput(self)

        else:
            print("ERROR Alle inputs schon voll")
        #print(str(self.inputs))

class And(Gate):

    def calcValue(self):
        return self.inputs[0].getValue() and self.inputs[1].getValue()


    def __init__(self,inputs=[None, None]):
        Gate.__init__(self,inputs)



class Or(Gate):

    def calcValue(self):
        return self.inputs[0].getValue() or self.inputs[1].getValue()

    def __init__(self,inputs=[None, None]):
        Gate.__init__(self,inputs)


class Nand(Gate):

    def calcValue(self):
        return not (self.inputs[0].getValue() and self.inputs[1].getValue())

    def __init__(self,inputs=[None, None]):
        Gate.__init__(self,inputs)


class Nor(Gate):

    def calcValue(self):
        return not (self.inputs[0].getValue() or self.inputs[1].getValue())

    def __init__(self,inputs=[None, None]):
        Gate.__init__(self,inputs)


class Not(Gate):

    def calcValue(self):
        return not self.inputs.getValue()

    def __init__(self,inputs= None):
        self.inputs = inputs
        Component.__init__(self)
        if not self.inputs== None:
            self.inputs.appendOutput(self)

    def addInput(self, input):
        if self.inputs == None:
            self.inputs = input
            self.inputs.appendOutput(self)
        else:
            print("ERROR Alle inputs schon voll")


class Input(Component):


    def __init__(self, input=False):
        Component.__init__(self,input)

    def calcValue(self):
        return self.getValue()
    def switch(self):
        self.value = not self.value
        self.update()
        return self.value
    def setFalse(self):
        self.value = False
        self.update()

class Output(Gate):

    def __init__(self, inputs=None):
        self.inputs = inputs
        Component.__init__(self)
        if not inputs== None:
            self.inputs.appendOutput(self)

        self.outputs = None
    def calcValue(self):
        return self.inputs.getValue()

    def addInput(self, input):
        if self.inputs == None:
            self.inputs = input
            self.inputs.appendOutput(self)
        else:
            print("ERROR Alle inputs schon voll")