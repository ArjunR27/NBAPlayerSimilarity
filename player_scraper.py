import requests
import csv
import time
from bs4 import BeautifulSoup


def main():
    url = 'https://www.basketball-reference.com/leagues/NBA_2024_totals.html'  # Update URL for 2023-2024 season
    response = requests.get(url)

    if response.status_code != 200:
        print(f'Status code: {response.status_code}')
        return
    
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    tds = soup.find_all('td', attrs={'data-stat': True})
    
    game_data = []
    columns = set()
    
    game_row = {}
    for td in tds:
        data_stat = td['data-stat']
        value = td.text.strip()
        
        if data_stat == 'awards' and game_row:
            game_data.append(game_row)
            columns.update(game_row.keys())
            game_row = {}
        
        game_row[data_stat] = value
    
    with open('player_data.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(columns))
        writer.writeheader()
        for row in game_data:
            writer.writerow(row)
    
    time.sleep(3)


if __name__ == "__main__":
    main()
