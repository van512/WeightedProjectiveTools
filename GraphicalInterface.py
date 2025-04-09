import tkinter as tk
from tkinter import messagebox
from WeightedProjTools import *


# Function to handle the calculation when the button is clicked
def calculate_dimension():
    try:
        a = list(map(int, weights_entry.get().split()))
        d = int(degree_entry.get())

        w = Weights(a)
        Cad = LinearSystem(w,d) #C_a[X]_d

        if Cad.isreducible == True:

            result_string=str(f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the wellformed weight {w.wellformed_weights}.\n"+
                          f"Thus we have the following equalities :\n"+
                          f"P({w.weights})=P({w.reduced_weights})=P({w.wellformed_weights}).\n \n"+
                          f"The dimension of the associated linear system of degree {d} is dim C_{w.weights}[X]_{d}={Cad.dim_before}.\n"+
                          f"The degree is reducible. \n"+ 
                          f"The reduced degree is {Cad.reduced_degree} and the corresponding wellformed degree is {Cad.wellformed_degree}.\n"+
                          f"We see that the dimensions of the linear systems agree: \n"+
                          f"dim C_{w.wellformed_weights}[X]_{Cad.wellformed_degree} = {Cad.dim_after}")
        else:
            result_string=str(f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the wellformed weight {w.wellformed_weights}.\n"+
                          f"Thus we have the following equalities :\n"+
                          f"P({w.weights})=P({w.reduced_weights})=P({w.wellformed_weights}).\n \n"+
                          f"The dimension of the associated linear system of degree {d} is dim C_{w.weights}[X]_{d}={Cad.dim_before}.\n"+
                          f"The degree is not reducible.")
        
        result_label.config(text=result_string, fg='black')
        




    except ValueError:
        result_label.config(text="Error: Please enter valid integers.", fg='red')

# Create the main window
root = tk.Tk()
root.title("Weight Reduction and Linear System Dimension Calculator")
root.geometry("700x400")
root.configure(bg='white')

# Create and place widgets
description_label = tk.Label(root, text="This program does the following :\n"
"- reduces the weights so they are well-formed                                                             \n"
"- calculates the dimension of the space of a-weighted homogeneous polynomials of degree d in n+1 variables\n"
"- reduces the degree when possible and calculates the dimension again                                     \n", bg='white')
description_label.pack(pady=10)

weights_label = tk.Label(root, text="Enter the weights (a_0, ..., a_n) separated by spaces:", bg='white')
weights_label.pack()
weights_entry = tk.Entry(root, width=50)
weights_entry.pack(pady=5)

degree_label = tk.Label(root, text="Enter the degree (d):", bg='white')
degree_label.pack()
degree_entry = tk.Entry(root, width=30)
degree_entry.pack(pady=5)

calculate_button = tk.Button(root, text="Calculate", command=calculate_dimension)
calculate_button.pack(pady=20)

result_label = tk.Label(root, text="", bg='white')
result_label.pack(pady=10)

# Run the application
root.mainloop()
