import csv

# --- Fonction pour faciliter les imports csv ---


def read_csv(filename):
    csvfile = open(filename)
    return csv.reader(csvfile, delimiter=';', quotechar='|')
