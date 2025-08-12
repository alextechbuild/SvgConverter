# SvgConverter.py
#
# Copyright 2025 Alex-Build
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.








# Libraries








import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
import cv2
import numpy as np
from skimage import measure
import svgwrite
from sklearn.cluster import KMeans








# Functionalities VectorizerApp








def image_to_colored_svg(image_path, output_svg_path, n_colors=10):
    # Charge l'image avec OpenCV (BGR)
    image = cv2.imread(image_path)
    # Convertit en RGB (car OpenCV charge en BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionne si image trop grande (pour limiter complexité)
    max_dim = 512
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    h, w = image.shape[:2]

    # Applatir pour KMeans (pixels x 3)
    img_reshaped = image.reshape(-1, 3)

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(img_reshaped)

    # Couleurs moyennes des clusters (entiers 0-255)
    colors = np.rint(kmeans.cluster_centers_).astype(int)

    # Création du dessin SVG avec la taille de l'image
    dwg = svgwrite.Drawing(output_svg_path, size=(w, h))

    # Pour chaque couleur / cluster
    for i, color in enumerate(colors):
        # Masque binaire des pixels appartenant au cluster i
        mask = (labels.reshape(h, w) == i).astype(np.uint8)

        # Trouve les contours (en pixels)
        contours = measure.find_contours(mask, 0.5)

        for contour in contours:
            # contour est une liste de (row, col) = (y, x)
            # On convertit en (x, y) pour SVG
            contour = np.flip(contour, axis=1)

            # Forme la chaîne SVG Path
            path_data = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in contour) + " Z"

            # Couleur en hex
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color)

            # Ajoute le chemin au SVG (rempli, sans contour)
            dwg.add(dwg.path(d=path_data, fill=color_hex, stroke="none"))

    # Sauvegarde le fichier SVG
    dwg.save()
    print(f"SVG enregistré dans : {output_svg_path}")








class VectorizerApp(QWidget):
    
    def __init__(self):
        
        super().__init__()
        self.setWindowTitle("Vectorisation Image -> SVG")
        self.layout = QVBoxLayout()

        self.label = QLabel("Choisissez une image à vectoriser")
        self.layout.addWidget(self.label)

        self.btn_open = QPushButton("Ouvrir une image")
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.btn_open)

        self.btn_convert = QPushButton("Vectoriser en SVG")
        self.btn_convert.clicked.connect(self.vectorize_image)
        self.btn_convert.setEnabled(False)
        self.layout.addWidget(self.btn_convert)

        self.setLayout(self.layout)
        self.image_path = None

    def open_file_dialog(self):
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            self.image_path = file_path
            self.label.setText(f"Image sélectionnée : {file_path}")
            self.btn_convert.setEnabled(True)

    def vectorize_image(self):
        
        if not self.image_path:
            
            QMessageBox.warning(self, "Erreur", "Aucune image sélectionnée.")
            return

        output_path = QFileDialog.getSaveFileName(self, "Enregistrer SVG sous", "output.svg", "SVG files (*.svg)")[0]
        
        if not output_path:
            return

        try:
            image_to_colored_svg(self.image_path, output_path)
            QMessageBox.information(self, "Succès", f"Vectorisation terminée !\nSVG enregistré sous :\n{output_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))








# Main Program








if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    window = VectorizerApp()
    window.resize(400, 150)
    window.show()
    sys.exit(app.exec_())
