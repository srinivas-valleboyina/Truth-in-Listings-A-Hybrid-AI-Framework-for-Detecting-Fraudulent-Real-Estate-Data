# main.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from ml_code import (
    load_dataset, preprocess_data, perform_eda, train_test_split_data,
    train_logistic_regression, train_knn_classifier, train_hybrid_extra_tree_ann,
    predict_class_only_real_estate
)

MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(exist_ok=True)

class App:
    def __init__(self, root):
        global screen_width, screen_height
        self.root = root
        self.root.title("Real Estate Listing Classifier")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.configure(bg="#f5f5f5")

        self.df = None
        self.df_processed = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text=" Real Estate Fake/Real Classifier",
                 font=("times", 20, "bold"), fg="#003366", bg="#f5f5f5").pack(pady=10)

        # Load background image (optional)
        if os.path.exists("real-estate-listings.jpg"):
            bg_img = Image.open("real-estate-listings.jpg").resize((screen_width, screen_height))
            self.bg_photo = ImageTk.PhotoImage(bg_img)
            canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
            canvas.pack(fill="both", expand=True)
            canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Main frame
        frame = tk.Frame(self.root, bg="white", bd=3, relief="groove", highlightbackground="#003366", highlightthickness=2)
        frame.place(relx=0.89, rely=0.23, anchor='n')
     
        

        # Buttons
        buttons = [
            ("Load Dataset", self.do_load),
            ("Preprocess Data", self.do_preprocess),
            ("Perform EDA", self.do_eda),
            ("Split Data", self.do_split),
            ("Train Logistic Regression", self.do_train_lr),
            ("Train KNN Classifier", self.do_train_knn),
            ("Train Hybrid ET + ANN", self.do_train_hybrid),
            ("Predict on New CSV", self.do_predict)
        ]
        for i, (text, command) in enumerate(buttons):
            tk.Button(
            frame, text=text, width=30, command=command,
            bg="#003366", fg="#ffffff", activebackground="#005599", activeforeground="#ffffff",
            font=("Helvetica", 11, "bold"), relief="raised", bd=2
        ).grid(row=i, column=0, padx=10, pady=4)

        # Output area
        self.output = scrolledtext.ScrolledText(
            self.root, width=110, height=25, font=("Consolas", 12), bg="#ffffff", fg="#000000", bd=2, relief="groove", insertbackground="#003366"
        )   
        self.output.place(relx=0.40, rely=0.23, anchor='n')

        # Image label for results
        self.image_label = tk.Label(self.root, bg="white")
        self.image_label.place(relx=0.5, rely=0.88, anchor='center')

    def show(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def show_plot(self, image_path):
        try:
            img = Image.open(image_path)
            img = img.resize((600, 400))
            self.plot_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.plot_img)
        except Exception as e:
            self.show(f"Error loading image: {e}")

    def do_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.df = load_dataset(path)
            self.show(f" Dataset loaded: {self.df}")

    def do_preprocess(self):
        if self.df is None:
            self.show(" Please load the dataset first.")
            return
        self.df_processed, self.X, self.y = preprocess_data(self.df, is_train=True)
        self.show(f" Data preprocessed. X shape: {self.X.shape}")

    def do_eda(self):
        if self.df is None:
            self.show(" Please load the dataset first.")
            return
        perform_eda(self.df)
        self.show("EDA completed and plots displayed.")
        #self.show_plot("eda_plots.png")

    def do_split(self):
        if self.X is None or self.y is None:
            self.show(" Please preprocess the data first.")
            return
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split_data(self.X, self.y)
        self.show(" Data split completed.")

    def do_train_lr(self):
        if self.X_train is None:
            self.show(" Please split the data first.")
            return
        result = train_logistic_regression(self.X_train, self.y_train, self.X_test, self.y_test)
        self.show("Logistics Regression model trained.")
        self.show(result)
       # self.show_plot("results/Logistic_Regression_confusion_matrix.png")

    def do_train_knn(self):
        if self.X_train is None:
            self.show(" Please split the data first.")
            return
        result = train_knn_classifier(self.X_train, self.y_train, self.X_test, self.y_test)
        self.show("KNN model trained.")
        self.show(result)
       # self.show_plot("results/KNN_Classifier_confusion_matrix.png")

    def do_train_hybrid(self):
        if self.X_train is None:
            self.show(" Please split the data first.")
            return
        result = train_hybrid_extra_tree_ann(self.X_train, self.y_train, self.X_test, self.y_test)
        self.show("Hybrid Extra Tree + ANN model trained.")
        self.show(result)
        #self.show_plot("results/Hybrid_AI_Model_confusion_matrix.png")

    def do_predict(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        test = pd.read_csv(path)
        test.fillna(test.mean(numeric_only=True), inplace=True)

        y_pred = predict_class_only_real_estate(test, model_name='hybrid_extra_tree_ann1.pkl')

        label_map = {0: 'Fake', 1: 'Real'}
        decoded = [label_map[int(label)] for label in y_pred]

        test['Predicted Label'] = decoded

        self.predicted_df = test  # <- Save the DataFrame in a class attribute

        self.show(" Predictions:\n")
        self.show(self.predicted_df.to_string(index=False))



        """# Just display in the GUI
        self.show("Prediction completed. Results:\n")
        for i, label in enumerate(decoded, start=1):
            self.show(f"Sample {i}: {label}")"""


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
