import tkinter as tk
from tkinter import IntVar
from tkinter import filedialog
import json
import os
from mentalStrategiesProject import runApp

def runGUI():
  print("Running GUI...")
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
        data_file_path_entry.delete(0, tk.END)  # Clear the entry field
        data_file_path_entry.insert(0, file_path)  # Insert selected directory path
  
  def validate_numeric_input(P):
    if P == "":
        return True
    try:
        float(P)
        return True
    except ValueError:
        return False

          
  app = tk.Tk()
  app.title("Variable Input GUI")

  success_cols_names_label = tk.Label(app, text="success columns names:")
  success_cols_names_entry = tk.Entry(app)
  
  subject_num_col_name_label = tk.Label(app, text="subject_number column name:")
  subject_num_col_name_entry = tk.Entry(app)

  session_num_col_name_label = tk.Label(app, text="session_number column name:")
  session_num_col_name_entry = tk.Entry(app)

  data_preprocessed_var = IntVar()
  data_preprocessed_box = tk.Checkbutton(app, text="data preprocessed?", variable=data_preprocessed_var)

  run_on_processed_data_var = IntVar()
  run_on_processed_data_box = tk.Checkbutton(app, text="run on the proccessed data (unchecked means run on raw data)", variable=run_on_processed_data_var)

  data_file_path_label = tk.Label(app, text="Select Raw Data File:")
  data_file_path_entry = tk.Entry(app, width=50)

  choose_data_file_path_button = tk.Button(app, text="Choose File", command=choose_directory)

  correlation_threshold_label = tk.Label(app, text="Enter Numeric Value:")
  correlation_threshold_entry = tk.Entry(app, validate="key", validatecommand=(app.register(validate_numeric_input), '%P'))


  def submit():

    subject_num_col_name = subject_num_col_name_entry.get()
    session_num_col_name = session_num_col_name_entry.get()
    data_preprocessed = data_preprocessed_var.get()
    run_on_processed_data = run_on_processed_data_var.get()
    correlation_threshold = correlation_threshold_entry.get()
    data_file_path = data_file_path_entry.get()
    success_cols_names = success_cols_names_entry.get()
      
    print("subject_num_col_name:", subject_num_col_name)
    print("session_num_col_name:", session_num_col_name)
    print("data_preprocessed:", data_preprocessed)
    print("run_on_processed_data:", run_on_processed_data)
    print("correlation_threshold:", correlation_threshold)
    print("data_file_path:", data_file_path)
    print("success_cols_names:", success_cols_names)

    variables_dict = {
      "success_cols_names": success_cols_names,
      "subject_num_col_name": subject_num_col_name,
      "session_num_col_name": session_num_col_name,
      "data_preprocessed": data_preprocessed,
      "run_on_processed_data": run_on_processed_data,
      "correlation_threshold": correlation_threshold,
      "data_file_path": data_file_path,
    }
    save_variables(variables_dict, "user_variables.json")  # Save variables to JSON file
    app.destroy()  # Close the GUI window

  submit_button = tk.Button(app, text="Submit", command=submit)


  data_file_path_label.pack(pady=5)
  data_file_path_entry.pack(pady=5)
  choose_data_file_path_button.pack()

  success_cols_names_label.pack(pady=5)
  success_cols_names_entry.pack()

  subject_num_col_name_label.pack()
  subject_num_col_name_entry.pack()

  session_num_col_name_label.pack()
  session_num_col_name_entry.pack()
  
  data_preprocessed_box.pack()
  run_on_processed_data_box.pack()

  correlation_threshold_label.pack()
  correlation_threshold_entry.pack()

  submit_button.pack(pady=10)

  app.mainloop()
