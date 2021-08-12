import xlrd
import difflib
import time
import csv
import pandas as pd
import smtplib, ssl

new_figs = []
old_figs = []
j = 1
port = 465
sender_email = "jamespythmail@gmail.com"
r_email = "james.pheby@afp.com"
receiver_email = "jamespheby@hotmail.com"

while True:
    print(j)
    read_file = pd.read_excel (r"https://www.arcgis.com/sharing/rest/content/items/bc8ee90225644ef7a6f4dd1b13ea1d67/data")
    read_file.to_csv (r'update.csv', index = None, header=True)
    with open('update.csv', mode='r') as latest_figs:
        csv_reader = csv.DictReader(latest_figs)
        for row in csv_reader:
            new_figs.append(row)
        if old_figs:
            if new_figs != old_figs:
                difference = [li for li in difflib.ndiff(new_figs[0], old_figs[0]) if li[0] != ' ']
                difference = str(difference)
                message = """\
                            Subject: Change csv website

                            """ + difference
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                    server.login("jamespythmail@gmail.com", "Waddler8")
                    server.sendmail(sender_email, receiver_email, message)
                    server.sendmail(sender_email, r_email, message)
                    server.quit()
        old_figs =[]
        old_figs = new_figs
        new_figs = []
        j += 1
        time.sleep(120)
