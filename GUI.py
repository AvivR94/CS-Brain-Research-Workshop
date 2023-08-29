import tkinter as tk
from tkinter import IntVar
from tkinter import filedialog
import json
import os

def saveToFile(filename, method='return path', content=None):
  script_dir = os.path.dirname(__file__)  # Directory of the script
  abs_file_path = os.path.join(script_dir, filename)
  if (method == 'json'):
    with open(abs_file_path, 'w') as file:
      json.dump(content, file)
  else:
     return abs_file_path


def runGUI():
  print("Running GUI...")
  # Save variables to JSON file

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

  num_of_sessions_label = tk.Label(app, text="Number of Sessions:")
  num_of_sessions_entry = tk.Entry(app)

  num_of_runs_per_session_label = tk.Label(app, text="Number of Runs per Session:")
  num_of_runs_per_session_entry = tk.Entry(app)

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

    num_of_sessions = num_of_sessions_entry.get()
    num_of_runs_per_session = num_of_runs_per_session_entry.get()
    subject_num_col_name = subject_num_col_name_entry.get()
    session_num_col_name = session_num_col_name_entry.get()
    data_preprocessed = data_preprocessed_var.get()
    run_on_processed_data = run_on_processed_data_var.get()
    correlation_threshold = correlation_threshold_entry.get()
    data_file_path = data_file_path_entry.get()
    success_cols_names = success_cols_names_entry.get()
      
    print("num_of_sessions:", num_of_sessions)
    print("num_of_runs_per_session:", num_of_runs_per_session)
    print("subject_num_col_name:", subject_num_col_name)
    print("session_num_col_name:", session_num_col_name)
    print("data_preprocessed:", data_preprocessed)
    print("run_on_processed_data:", run_on_processed_data)
    print("correlation_threshold:", correlation_threshold)
    print("data_file_path:", data_file_path)
    print("success_cols_names:", success_cols_names)

    variables_dict = {
      "num_of_sessions": num_of_sessions,
      "num_of_runs_per_session": num_of_runs_per_session,
      "success_cols_names": success_cols_names,
      "subject_num_col_name": subject_num_col_name,
      "session_num_col_name": session_num_col_name,
      "data_preprocessed": data_preprocessed,
      "run_on_processed_data": run_on_processed_data,
      "correlation_threshold": correlation_threshold,
      "data_file_path": data_file_path,
    }
    saveToFile("user_variables.json", 'json', variables_dict)  # Save variables to JSON file
    app.destroy()  # Close the GUI window

  submit_button = tk.Button(app, text="Submit", command=submit)


  data_file_path_label.pack(pady=5)
  data_file_path_entry.pack(pady=5)
  choose_data_file_path_button.pack()

  num_of_sessions_label.pack(pady=5)
  num_of_sessions_entry.pack()

  num_of_runs_per_session_label.pack()
  num_of_runs_per_session_entry.pack()

  success_cols_names_label.pack()
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
# runGUI()