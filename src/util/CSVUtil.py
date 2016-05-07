import sys
import csv

def WriteCSV(fileName,data):
    csvfile = file(fileName, 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(data)
    csvfile.close()

def ReadCSV(fileName):
    csvfile = file(fileName, 'rb')
    reader = csv.reader(csvfile)
    content = [item for item in reader]
    reader.close()
    return content;