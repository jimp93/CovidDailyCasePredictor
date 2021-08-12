import requests
from bs4 import BeautifulSoup
import urllib
import smtplib, ssl
import xlrd
import difflib
import time
import csv
import pandas as pd

i = 1
old_text = []
new_text = []
new_figs = []
old_figs = []
port = 465
sender_email = "jamespythmail@gmail.com"
receiver_email = "jamespheby@hotmail.com"
r_email = "james.pheby@afp.com"
g_email = "ldn.news@afp.com"


while True:
    print(i)
    indicator = str
    url = "https://www.gov.uk/guidance/coronavirus-covid-19-information-for-the-public"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    new_text.append(text)
    if old_text:
        if new_text != old_text:
            # difference = [li for li in difflib.ndiff(new_text[0], old_text[0]) if li[0] != ' ']
            # difference = str(difference)
            message = """\From: From Person <jamespythmail@gmail.com>
            To: To Person <james.pheby@afp.com>
            Subject: Change to gov website

            The gov website has been updated
            """
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                server.login("jamespythmail@gmail.com", "Waddler8")
                server.sendmail(sender_email, receiver_email, message)
                server.sendmail(sender_email, r_email, message)
                server.sendmail(sender_email, g_email, message)
                server.quit()
    old_text = []
    old_text.append(text)
    new_text = []

    read_file = pd.read_excel(
        r"https://www.arcgis.com/sharing/rest/content/items/bc8ee90225644ef7a6f4dd1b13ea1d67/data")
    read_file.to_csv(r'update.csv', index=None, header=True)
    with open('update.csv', mode='r') as latest_figs:
        csv_reader = csv.DictReader(latest_figs)
        for row in csv_reader:
            new_figs.append(row)
        if old_figs:
            if new_figs != old_figs:
                # cdifference = [cli for cli in difflib.ndiff(new_figs[0], old_figs[0]) if cli[0] != ' ']
                # cdifference = str(cdifference)
                cmessage = """\From: From Person <jamespythmail@gmail.com>
                To: To Person <james.pheby@afp.com>
                Subject: Change to csv website

                The csv has been updated
                """
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
                    server.login("jamespythmail@gmail.com", "Waddler8")
                    server.sendmail(sender_email, receiver_email, cmessage)
                    server.sendmail(sender_email, r_email, cmessage)
                    server.sendmail(sender_email, g_email, cmessage)
                    server.quit()
        old_figs = []
        old_figs = new_figs
        new_figs = []
    i += 1
    time.sleep(120)
