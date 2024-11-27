import requests
import csv
import time
import re
from bs4 import BeautifulSoup

def main(url, header_written, year):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return
    
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    rows = soup.find_all('tr', class_=lambda x: x != 'thead')
    
    player_data = []
    columns = set()
    multi_team_players = set()

    for row in rows:
        player_row = {}
        player_name = None
        is_valid_row = True 

        for td in row.find_all('td', attrs={'data-stat': True}):
            data_stat = td['data-stat']
            value = td.text.strip()

            if data_stat == 'name_display' and value:
                if "League Average" in value:
                    is_valid_row = False
                    break

                player_name = value
                if player_name in multi_team_players:
                    break

            if data_stat == 'team_name_abbr' and value and bool(re.match(r"^[2-9]TM$", value)):
                multi_team_players.add(player_name)

            if data_stat == 'games' and value and int(value) < 41:
                is_valid_row = False
                break

            player_row[data_stat] = value
            columns.add(data_stat)
        
        if is_valid_row and player_row:
            player_data.append(player_row)
    
    # Save data to CSV
    with open(f'./season_data/player_data_{year}.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(columns))
        if not header_written:
            writer.writeheader()
            header_written = True  # Ensure the header is written only once
        for row in player_data:
            writer.writerow(row)
    
    time.sleep(3)

if __name__ == "__main__":
    for year in range(2000, 2025):
        main(url=f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html", header_written=False, year=year)
