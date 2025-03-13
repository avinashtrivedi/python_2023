import unittest
from world_cup_scoreboard import Scoreboard

class TestWorldCupScoreboard(unittest.TestCase):
    def setUp(self):
        # Create a new Scoreboard instance for each test case
        self.scoreboard = Scoreboard()

    def test_start_match(self):
        # Test the 'start_match' method of the Scoreboard class
        self.scoreboard.start_match("Mexico", "Canada")
        # Retrieve the match by index and verify its attributes
        match = self.scoreboard.get_match_by_index(0)
        self.assertEqual(match.home_team, "Mexico")
        self.assertEqual(match.away_team, "Canada")
        self.assertEqual(match.home_score, 0)
        self.assertEqual(match.away_score, 0)

    def test_update_match_score(self):
        # Test the 'update_match_score' method of the Scoreboard class
        self.scoreboard.start_match("Spain", "Brazil")
        # Update the match scores and verify the changes
        self.scoreboard.update_match_score(0, 10, 2)
        match = self.scoreboard.get_match_by_index(0)
        self.assertEqual(match.home_score, 10)
        self.assertEqual(match.away_score, 2)

    def test_finish_match(self):
        # Test the 'finish_match' method of the Scoreboard class
        self.scoreboard.start_match("Argentina", "Australia")
        # Finish the match and verify that it is removed from the scoreboard
        self.scoreboard.finish_match(0)
        self.assertEqual(len(self.scoreboard.matches), 0)

    def test_get_summary(self):
        # Test the 'get_summary' method of the Scoreboard class
        self.scoreboard.start_match("Mexico", "Canada")
        self.scoreboard.update_match_score(0, 0, 5)

        self.scoreboard.start_match("Spain", "Brazil")
        self.scoreboard.update_match_score(1, 10, 2)

        self.scoreboard.start_match("Germany", "France")
        self.scoreboard.update_match_score(2, 2, 2)

        self.scoreboard.start_match("Uruguay", "Italy")
        self.scoreboard.update_match_score(3, 6, 6)

        self.scoreboard.start_match("Argentina", "Australia")
        self.scoreboard.update_match_score(4, 3, 1)

        # Get the summary and verify its correctness
        summary = self.scoreboard.get_summary()
        self.assertEqual(len(summary), 5)
        self.assertEqual(summary[0].home_team, "Uruguay")
        self.assertEqual(summary[1].home_team, "Spain")
        self.assertEqual(summary[4].home_team, "Germany")

if __name__ == "__main__":
    unittest.main()
