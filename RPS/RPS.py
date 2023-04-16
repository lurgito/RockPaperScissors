import random
from sklearn import tree

# Define the possible moves and their corresponding numeric values
moves = {
  "rock": 0,
  "paper": 1,
  "scissors": 2
}

# Define the winning combinations
winning_combinations = {
  "rock": "scissors",
  "paper": "rock",
  "scissors": "paper"
}

# Define the data set for training the machine learning model
data_set = [
  [0, 1, 1], # Rock beats scissors
  [1, 2, 1], # Paper beats rock
  [2, 0, 1], # Scissors beat paper
  [1, 0, -1], # Paper loses to rock
  [2, 1, -1], # Scissors lose to paper
  [0, 2, -1] # Rock loses to scissors
]

# Define the machine learning model
clf = tree.DecisionTreeClassifier()

# Train the model with the data set
X = [[row[0], row[1]] for row in data_set]
y = [row[2] for row in data_set]
clf = clf.fit(X, y)

# Define the function for playing the game
def play_game():
  # Get the player's move
  player_move = input("Enter your move (rock/paper/scissors): ").lower()
  
  # Validate the player's move
  if player_move not in moves:
    print("Invalid move. Please try again.")
    play_game()

  # Get the computer's move
  computer_move = random.choice(list(moves.keys()))

  # Determine the winner using the machine learning model
  winner = clf.predict([[moves[player_move], moves[computer_move]]])[0]
  if winner == 1:
    print(f"You win! {player_move.capitalize()} beats {computer_move.capitalize()}.")
  elif winner == -1:
    print(f"You lose! {computer_move.capitalize()} beats {player_move.capitalize()}.")
  else:
    print(f"It's a tie! You both chose {player_move.capitalize()}.")

  # Ask the player if they want to play again
  play_again = input("Do you want to play again? (y/n)").lower()
  if play_again == "y":
    play_game()
  else:
    print("Thanks for playing!")

# Start the game
play_game()
