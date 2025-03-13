# Represents an individual match in the World Cup
class Match:
    def __init__(self, home_team, away_team, SrNum):
        # Initialize match attributes
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.SrNum = SrNum

    # Update the scores for the match
    def update_score(self, home_score, away_score):
        self.home_score = home_score
        self.away_score = away_score

    # Calculate and return the total score of the match
    def total_score(self):
        return self.home_score + self.away_score


# Manages the live World Cup Scoreboard
class Scoreboard:
    def __init__(self):
        # Initialize the list to store ongoing matches and match number counter
        self.matches = []
        self.sr_num = 0

    # Start a new match and add it to the scoreboard
    def start_match(self, home_team, away_team):
        # Increment match number counter
        self.sr_num = self.sr_num + 1
        # Create a new match instance with the provided teams and match number
        match = Match(home_team, away_team, self.sr_num)
        # Append the match to the list of ongoing matches
        self.matches.append(match)

    # Finish a match currently in progress and remove it from the scoreboard
    def finish_match(self, index):
        if 0 <= index < len(self.matches):
            # Remove the match at the specified index from the list of ongoing matches
            del self.matches[index]

    # Update the scores for a match in progress
    def update_match_score(self, index, home_score, away_score):
        if 0 <= index < len(self.matches):
            # Get the match at the specified index and update its scores
            match = self.matches[index]
            match.update_score(home_score, away_score)
            
    def get_match_by_index(self, index):
        if 0 <= index < len(self.matches):
            return self.matches[index]
        

    # Get a summary of matches in progress ordered by their total score and start time
    def get_summary(self):
        sorted_matches = sorted(self.matches, key=lambda match: (-match.total_score(), -match.SrNum))
        return sorted_matches
    
if __name__ == "__main__":
    scoreboard = Scoreboard()

    scoreboard.start_match("Mexico", "Canada")
    scoreboard.update_match_score(0, 0, 5)

    scoreboard.start_match("Spain", "Brazil")
    scoreboard.update_match_score(1, 10, 2)

    scoreboard.start_match("Germany", "France")
    scoreboard.update_match_score(2, 2, 2)

    scoreboard.start_match("Uruguay", "Italy")
    scoreboard.update_match_score(3, 6, 6)

    scoreboard.start_match("Argentina", "Australia")
    scoreboard.update_match_score(4, 3, 1)

    summary = scoreboard.get_summary()

    # Print the summary of matches in descending order of total score and start time
    for idx, match in enumerate(summary, start=1):
        print(f"{idx}. {match.home_team} {match.home_score} - {match.away_team} {match.away_score}")