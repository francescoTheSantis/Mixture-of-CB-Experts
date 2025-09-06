import sympy, torch, sympytorch

"""
The code is adapted from: https://github.com/patrick-kidger/sympytorch/tree/master.
This script shows an example of using SymPy with PyTorch by using the sympytorch library.
By using this library, it is possible to convert a textual string into an executable PyTorch module, 
which can be used for automatic differentiation and optimization.
"""

### 1. Define the symbols (variables)
x, y = sympy.symbols('x y')

### 2. Define the equation in a textual format
exp = "1.0 * cos(x) + exp(y) + x/y"

### 3. Convert to sympy expression
exp = sympy.sympify(exp)

### 4. Convert the textual equation into an executable PyTorch module
torch_exp = sympytorch.SymPyModule(expressions=[exp])

### 5. Execute the module
x_ = torch.rand(3, requires_grad=True)
y_ = torch.rand(3, requires_grad=True)
out = torch_exp(x=x_, y=y_)
y = out.mean()
y.backward()
print(x_.grad)

### 6. How to create a memory of functions and train a selector to choose between them
x, y = sympy.symbols('x y')

exp1 = "1+x"
exp2 = "x-y"
exp3 = "x+y"

x_ = torch.tensor([0.0,1.0,2.0,3.0], requires_grad=True).unsqueeze(0).expand(10,-1) # (bsz, n_vars)
y_ = torch.tensor([0.0,1.0,2.0,3.0], requires_grad=True).unsqueeze(0).expand(10,-1)

functions = []
for exp in [exp1, exp2, exp3]:
    torch_exp = sympytorch.SymPyModule(expressions=[sympy.sympify(exp)])
    functions.append(torch_exp)

selector_output = torch.tensor([0.1, 0.2, 0.7], requires_grad=True)

for i, func in enumerate(functions):
    output = func(x=x_, y=y_)
    # Combine the outputs using the selector
    output = output * selector_output[i]

output = output.mean()

output.backward()

print(x_.grad, y_.grad, selector_output.grad)

### 7. Train a selector that uses monte carlo sampling
x, y = sympy.symbols('x y')

exp1 = "1+x"
exp2 = "x-y"
exp3 = "x+y"

x_ = torch.tensor([0.0,1.0,2.0,3.0], requires_grad=True)
y_ = torch.tensor([0.0,1.0,2.0,3.0], requires_grad=True)

functions = []
for exp in [exp1, exp2, exp3]:
    torch_exp = sympytorch.SymPyModule(expressions=[sympy.sympify(exp)])
    functions.append(torch_exp)

selection_tensor = torch.where(torch.softmax(torch.randn(4, 3, 1, 10), dim=1)>0.5, 1.0, 0.0) # (bsz, n_functions, n_classes, n_samples)
selection_tensor.requires_grad=True

outputs = []
for i, func in enumerate(functions):
    output = func(x=x_, y=y_)
    outputs.append(output)

outputs = torch.stack(outputs, dim=1).squeeze() # (bsz, n_functions)

# Combine the outputs using the selector
output = torch.einsum('bfcs,bf->bcs', selection_tensor, outputs)

output = output.mean()
output.backward()

print("Gradients propagate through: x, y and the selector")
print(x_.grad, y_.grad, selection_tensor.grad)