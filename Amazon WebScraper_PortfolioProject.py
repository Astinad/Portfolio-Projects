# -*- coding: utf-8 -*-
"""Amazon WebScraper_PortfolioProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xQR7rCx1BrWGMHOWLYru3abTn-cCQl3u
"""

from bs4 import BeautifulSoup
import requests
import time
import datetime
import smtplib

URL = 'https://www.amazon.com/HyperX-Cloud-Gaming-Headset-KHX-HSCP-GM/dp/B00SAYCVTQ/ref=sr_1_4?keywords=gaming+headsets&pd_rd_r=a4796646-0121-426f-964b-70bbbe1aef88&pd_rd_w=8cj1v&pd_rd_wg=GoDSW&pf_rd_p=12129333-2117-4490-9c17-6d31baf0582a&pf_rd_r=8EQ5FNTZPGY00HWRRP54&qid=1678133511&sr=8-4'
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
page = requests.get(URL, headers=headers)

soup1 = BeautifulSoup(page.content, "html.parser")
soup2 = BeautifulSoup(soup1.prettify(), "html.parser")

title = soup2.find(id='title_feature_div').get_text(separator=" ")
price = soup2.find('span', {'class':"a-price a-text-price a-size-medium apexPriceToPay"}).text.strip()

print(title)
print(price)

#Removing space
price = price.strip()[1:]
title = title.strip()

print(title)
print(price)

#Create a Timestamp for your output to track when data was collected
today = datetime.date.today()
print(today)

# Create CSV and write headers and data into the file
import csv 

header = ['Title', 'Price', 'Date']
data = [title, price, today]

with open('AmazonWebScraperDataset.csv', 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(data)

import pandas as pd
df = pd.read_csv(r'C:\Users\Admin\AmazonWebScraperDataset.csv')
print(df)

#Append data to the csv file
with open('AmazonWebScraperDataset.csv', 'a+', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(data)

#Combine all of the above code into one function


def check_price():
    URL = 'https://www.amazon.com/HyperX-Cloud-Gaming-Headset-KHX-HSCP-GM/dp/B00SAYCVTQ/ref=sr_1_4?keywords=gaming+headsets&pd_rd_r=a4796646-0121-426f-964b-70bbbe1aef88&pd_rd_w=8cj1v&pd_rd_wg=GoDSW&pf_rd_p=12129333-2117-4490-9c17-6d31baf0582a&pf_rd_r=8EQ5FNTZPGY00HWRRP54&qid=1678133511&sr=8-4'
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    
    page = requests.get(URL, headers=headers)

    soup1 = BeautifulSoup(page.content, "html.parser")

    soup2 = BeautifulSoup(soup1.prettify(), "html.parser")

    title = soup2.find(id='title_feature_div').get_text()
    
    price = soup2.find('span', {'class':"a-price a-text-price a-size-medium apexPriceToPay"}).text.strip()

    price = price.strip()[1:]
    title = title.strip()

    import datetime

    today = datetime.date.today()
    
    import csv 

    header = ['Title', 'Price', 'Date']
    data = [title, price, today]

    with open('AmazonWebScraperDataset.csv', 'a+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

# Runs check_price after a set time and inputs data into your CSV
while(True):
    check_price()
    time.sleep(86400)

import pandas as pd
df = pd.read_csv(r'C:\Users\Admin\AmazonWebScraperDataset.csv')
print(df)

# Send yourself an email when a price hits below a certain level

def send_mail():
    server = smtplib.SMTP_SSL('smtp.gmail.com',465)
    server.ehlo()
    #server.starttls()
    server.ehlo()
    server.login('astinad@gmail.com','xxxxxxxxxxx')
    
    subject = "The headphones you want is below $80! Now is your chance to buy!"
    body = "Astina, This is the moment we have been waiting for. Now is your chance to pick up the headphones of your dreams. Don't miss it! Link here: https://www.amazon.com/HyperX-Cloud-Gaming-Headset-KHX-HSCP-GM/dp/B00SAYCVTQ/ref=sr_1_4?keywords=gaming+headsets&pd_rd_r=a4796646-0121-426f-964b-70bbbe1aef88&pd_rd_w=8cj1v&pd_rd_wg=GoDSW&pf_rd_p=12129333-2117-4490-9c17-6d31baf0582a&pf_rd_r=8EQ5FNTZPGY00HWRRP54&qid=1678133511&sr=8-4"
   
    msg = f"Subject: {subject}\n\n{body}"
    
    server.sendmail('astinad@gmail.com', msg)