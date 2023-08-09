import tkinter as tk
from tkinter import IntVar
from tkinter import filedialog
import json
import os
from mentalStrategiesProject import runApp

def runGUI():
  # Save variables to JSON file
  def save_variables(variables_dict, filename):
    script_dir = os.path.dirname(__file__)  # Directory of the script
    abs_file_path = os.path.join(script_dir, filename)

    with open(abs_file_path, 'w') as file:
      json.dump(variables_dict, file)
    # with open(filename, 'w') as file:
    #   json.dump(variables_dict, file)

  def choose_directory():
    file_path = filedialog.askopenfilename()  # Open directory selection dialog
    if file_path:
        directory_entry.delete(0, tk.END)  # Clear the entry field
        directory_entry.insert(0, file_path)  # Insert selected directory path

          
  app = tk.Tk()
  app.title("Variable Input GUI")

  label_var1 = tk.Label(app, text="Variable 1:")
  entry_var1 = tk.Entry(app)

  label_var2 = tk.Label(app, text="Variable 2:")
  entry_var2 = tk.Entry(app)

  var_option1 = IntVar()
  check_option1 = tk.Checkbutton(app, text="Option 1", variable=var_option1)

  var_option2 = IntVar()
  check_option2 = tk.Checkbutton(app, text="Option 2", variable=var_option2)

  directory_label = tk.Label(app, text="Selected Directory:")
  directory_entry = tk.Entry(app, width=50)

  choose_button = tk.Button(app, text="Choose Directory", command=choose_directory)

  def submit():
    variable1 = entry_var1.get()
    variable2 = entry_var2.get()
    option1 = var_option1.get()
    option2 = var_option2.get()
      
    print("Variable 1:", variable1)
    print("Variable 2:", variable2)
    print("Option 1:", option1)
    print("Option 2:", option2)

    variables_dict = {
      "variable1": variable1,
      "variable2": variable2,
      "option1": option1,
      "option2": option2
    }
    save_variables(variables_dict, "user_variables.json")  # Save variables to JSON file
    app.destroy()  # Close the GUI window

  submit_button = tk.Button(app, text="Submit", command=submit)

  label_var1.pack()
  entry_var1.pack()

  label_var2.pack()
  entry_var2.pack()

  check_option1.pack()
  check_option2.pack()

  directory_label.pack(pady=5)
  directory_entry.pack(pady=5)
  choose_button.pack(pady=5)

  submit_button.pack()

  app.mainloop()


