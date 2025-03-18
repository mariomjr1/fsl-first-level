import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox
import nipype.pipeline.engine as pe
from nipype.algorithms import modelgen
from nipype.interfaces import fsl
from nipype.interfaces.base import Bunch
import pandas as pd
import pickle, gzip
import nilearn.plotting, nilearn.image
from bids.layout import BIDSLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSL Analysis GUI")
        self.setGeometry(100, 100, 600, 400)

        # Layout
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # BIDS Folder
        self.btn_bids = QPushButton("Choose BIDS Folder")
        self.lbl_bids = QLabel("No folder selected")
        self.btn_bids.clicked.connect(self.choose_bids)
        layout.addWidget(self.btn_bids)
        layout.addWidget(self.lbl_bids)

        # Output Folder
        self.btn_output = QPushButton("Choose Output Folder")
        self.lbl_output = QLabel("No folder selected")
        self.btn_output.clicked.connect(self.choose_output)
        layout.addWidget(self.btn_output)
        layout.addWidget(self.lbl_output)

        # Task Names
        layout.addWidget(QLabel("Task Names (comma-separated):"))
        self.txt_tasks = QLineEdit("Finger, Foot, Lips")
        layout.addWidget(self.txt_tasks)

        # Session
        layout.addWidget(QLabel("Session:"))
        self.txt_session = QLineEdit("test")
        layout.addWidget(self.txt_session)

        # Contrasts
        layout.addWidget(QLabel("Contrasts (e.g., 'Name: Task1=1, Task2=-1'):"))
        self.txt_contrasts = QTextEdit("Finger vs. Rest: Finger=1\nAll Motor: Finger=1, Foot=1, Lips=1")
        layout.addWidget(self.txt_contrasts)

        # Confounds
        layout.addWidget(QLabel("Confound Columns:"))
        self.combo_confounds = QComboBox()
        self.combo_confounds.setEnabled(False)
        layout.addWidget(self.combo_confounds)

        # Buttons
        self.btn_first = QPushButton("Run First-Level Analysis")
        self.btn_second = QPushButton("Run Second-Level Analysis")
        self.btn_first.clicked.connect(self.run_first_level)
        self.btn_second.clicked.connect(self.run_second_level)
        layout.addWidget(self.btn_first)
        layout.addWidget(self.btn_second)

        # Status
        self.status = QTextEdit("Ready")
        self.status.setReadOnly(True)
        layout.addWidget(self.status)

        # Variables
        self.bids_dir = ""
        self.output_dir = ""
        self.layout = None

    def log(self, message):
        self.status.append(message)

    def choose_bids(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose BIDS Folder")
        if folder:
            self.bids_dir = folder
            self.lbl_bids.setText(os.path.basename(folder))
            try:
                self.log("Step 1: Loading BIDS layout...")
                self.layout = BIDSLayout(self.bids_dir)
                confounds_file = self.layout.get(suffix="confounds", extension=[".tsv", ".csv"])[0].path
                confounds = pd.read_csv(confounds_file, sep="\t")
                self.combo_confounds.clear()
                self.combo_confounds.addItems(confounds.columns)
                self.combo_confounds.setEnabled(True)
                self.log("BIDS layout loaded successfully.")
            except Exception as e:
                self.log(f"Error in Step 1 (Load BIDS): {str(e)} - File: {confounds_file}")

    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if folder:
            self.output_dir = folder
            self.lbl_output.setText(os.path.basename(folder))

    def create_feat_first_level_wf(self, name="Feat1stLevel"):
        # Same as original, unchanged for brevity
        feat_wf = pe.Workflow(name=name)
        # [Your workflow definition here]
        return feat_wf

    def parse_contrasts(self):
        contrasts = []
        for line in self.txt_contrasts.toPlainText().split("\n"):
            if not line.strip():
                continue
            name, spec = line.split(":", 1)
            weights = {}
            for pair in spec.split(","):
                task, weight = pair.split("=")
                weights[task.strip()] = float(weight.strip())
            contrasts.append([name.strip(), "T", list(weights.keys()), list(weights.values())])
        return contrasts

    def run_first_level(self):
        if not self.bids_dir or not self.output_dir:
            QMessageBox.warning(self, "Error", "Select BIDS and output folders!")
            return

        self.log("Step 2: Setting up first-level analysis...")
        try:
            tasks = [t.strip() for t in self.txt_tasks.text().split(",")]
            session = self.txt_session.text().strip()
            contrasts = self.parse_contrasts()
            confound_cols = [self.combo_confounds.itemText(i) for i in range(self.combo_confounds.count()) if self.combo_confounds.itemText(i) in self.combo_confounds.currentText()]

            bold_file = self.layout.get(suffix="bold", task=tasks[0], session=session, extension="nii.gz")[0].path
            mask_file = self.layout.get(suffix="brainmask", task=tasks[0], session=session, extension="nii.gz")[0].path
            events_file = self.layout.get(suffix="events", task=tasks[0], session=session, extension="tsv")[0].path
            confounds_file = self.layout.get(suffix="confounds", extension="tsv")[0].path

            events = pd.read_csv(events_file, sep="\t")
            confounds = pd.read_csv(confounds_file, sep="\t")
            info = [Bunch(
                conditions=tasks,
                onsets=[list(events[events.trial_type == t].onset - 10) for t in tasks],
                durations=[list(events[events.trial_type == t].duration) for t in tasks],
                regressors=[list(confounds[c].fillna(0)[4:]) for c in confound_cols],
                regressor_names=confound_cols
            )]

            feat_wf = self.create_feat_first_level_wf()
            feat_wf.base_dir = self.output_dir
            feat_wf.inputs.inputSource.in_func = bold_file
            feat_wf.inputs.inputSource.brain_mask = mask_file
            feat_wf.inputs.inputSource.subject_info = info
            feat_wf.inputs.inputSource.contrasts = contrasts

            self.log("Step 3: Running first-level analysis...")
            feat_wf.run()

            self.log("Step 4: Saving results...")
            filmgls_results = pickle.load(gzip.open(f"{self.output_dir}/Feat1stLevel/modelestimate/result_modelestimate.pklz", "rb"))
            for i, t_map in enumerate(filmgls_results.outputs.zstats[0]):
                plot = nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(t_map, 8), threshold=2.3)
                plot.savefig(f"{self.output_dir}/zstat_{i}.png")
                plot.close()
            self.log("First-level analysis completed!")

        except Exception as e:
            step = "Unknown"
            file = "Unknown"
            if "Step 2" in self.status.toPlainText():
                step = "Setup Workflow"
                file = bold_file if "bold" in str(e) else events_file if "events" in str(e) else confounds_file
            elif "Step 3" in self.status.toPlainText():
                step = "Run Analysis"
                file = f"{self.output_dir}/Feat1stLevel"
            elif "Step 4" in self.status.toPlainText():
                step = "Save Results"
                file = f"{self.output_dir}/zstat_*.png"
            self.log(f"Error in {step}: {str(e)} - File: {file}")

    def run_second_level(self):
        self.log("Second-level analysis not implemented yet.")  # Placeholder

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
