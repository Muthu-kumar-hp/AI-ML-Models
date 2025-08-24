import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import random
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

class MLRockPaperScissors:
    def __init__(self, sequence_length=5, model_type='RandomForest'):
        """
        Initialize the ML Rock Paper Scissors game
        
        Args:
            sequence_length: Number of previous moves to consider for prediction
            model_type: Type of ML model ('RandomForest', 'LogisticRegression', 'DecisionTree')
        """
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.moves = ['Rock', 'Paper', 'Scissors']
        self.move_to_beat = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
        
        # Game history
        self.player_history = deque(maxlen=1000)  # Store last 1000 moves
        self.game_data = []
        
        # Scores
        self.player_score = 0
        self.computer_score = 0
        self.ties = 0
        
        # ML Model
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.moves)
        self.is_trained = False
        
        # Model file path
        self.model_file = f'rps_model_{model_type.lower()}.joblib'
        self.history_file = 'rps_history.joblib'
        
        # Initialize model
        self._initialize_model()
        self._load_model_and_history()
    
    def _initialize_model(self):
        """Initialize the ML model based on the specified type"""
        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        else:
            raise ValueError("Model type must be 'RandomForest', 'LogisticRegression', or 'DecisionTree'")
    
    def _load_model_and_history(self):
        """Load saved model and game history if they exist"""
        try:
            if os.path.exists(self.model_file):
                self.model = joblib.load(self.model_file)
                self.is_trained = True
                print(f"Loaded trained {self.model_type} model from {self.model_file}")
            
            if os.path.exists(self.history_file):
                history_data = joblib.load(self.history_file)
                self.player_history = deque(history_data['player_history'], maxlen=1000)
                self.player_score = history_data.get('player_score', 0)
                self.computer_score = history_data.get('computer_score', 0)
                self.ties = history_data.get('ties', 0)
                print(f"Loaded game history: {len(self.player_history)} previous moves")
                
        except Exception as e:
            print(f"Could not load saved data: {e}")
    
    def _save_model_and_history(self):
        """Save the trained model and game history"""
        try:
            if self.is_trained:
                joblib.dump(self.model, self.model_file)
            
            history_data = {
                'player_history': list(self.player_history),
                'player_score': self.player_score,
                'computer_score': self.computer_score,
                'ties': self.ties
            }
            joblib.dump(history_data, self.history_file)
            
        except Exception as e:
            print(f"Could not save data: {e}")
    
    def _create_features(self, history):
        """Create feature vectors from move history"""
        if len(history) < self.sequence_length:
            # Pad with random moves if not enough history
            padded_history = [random.choice(self.moves) for _ in range(self.sequence_length - len(history))]
            padded_history.extend(history)
            history = padded_history
        
        # Convert moves to numbers
        encoded_moves = self.label_encoder.transform(history[-self.sequence_length:])
        return encoded_moves
    
    def _train_model(self):
        """Train the ML model on available data"""
        if len(self.player_history) < self.sequence_length + 1:
            return False
        
        X, y = [], []
        
        # Create training data from history
        history_list = list(self.player_history)
        for i in range(len(history_list) - self.sequence_length):
            features = self._create_features(history_list[:i + self.sequence_length])
            target = history_list[i + self.sequence_length]
            X.append(features)
            y.append(target)
        
        if len(X) > 0:
            X = np.array(X)
            y = self.label_encoder.transform(y)
            
            self.model.fit(X, y)
            self.is_trained = True
            return True
        
        return False
    
    def predict_next_move(self):
        """Predict the player's next move"""
        if not self.is_trained or len(self.player_history) < self.sequence_length:
            return random.choice(self.moves)
        
        try:
            features = self._create_features(list(self.player_history)).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            predicted_move = self.label_encoder.inverse_transform([prediction])[0]
            return predicted_move
        except:
            return random.choice(self.moves)
    
    def get_computer_move(self, predicted_player_move):
        """Get computer's move to beat the predicted player move"""
        # Add some randomness to avoid being too predictable
        if random.random() < 0.1:  # 10% chance of random move
            return random.choice(self.moves)
        return self.move_to_beat[predicted_player_move]
    
    def determine_winner(self, player_move, computer_move):
        """Determine the winner of the round"""
        if player_move == computer_move:
            return "Tie"
        elif self.move_to_beat[computer_move] == player_move:
            return "Player"
        else:
            return "Computer"
    
    def play_round(self, player_move):
        """Play a single round"""
        # Predict next move before adding current move to history
        predicted_move = self.predict_next_move()
        
        # Get computer move
        computer_move = self.get_computer_move(predicted_move)
        
        # Determine winner
        winner = self.determine_winner(player_move, computer_move)
        
        # Update scores
        if winner == "Player":
            self.player_score += 1
        elif winner == "Computer":
            self.computer_score += 1
        else:
            self.ties += 1
        
        # Add player move to history
        self.player_history.append(player_move)
        
        # Retrain model with new data
        if len(self.player_history) >= self.sequence_length + 1:
            self._train_model()
        
        # Save progress
        self._save_model_and_history()
        
        return {
            'player_move': player_move,
            'computer_move': computer_move,
            'predicted_move': predicted_move,
            'winner': winner,
            'player_score': self.player_score,
            'computer_score': self.computer_score,
            'ties': self.ties
        }

class RPSGameGUI:
    def __init__(self):
        self.game = MLRockPaperScissors()
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("ML Rock Paper Scissors")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🤖 ML Rock Paper Scissors 🤖", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=10)
        
        # Info label
        info_label = tk.Label(
            self.root,
            text="The AI learns from your moves to predict your next choice!",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666'
        )
        info_label.pack(pady=5)
        
        # Move buttons frame
        buttons_frame = tk.Frame(self.root, bg='#f0f0f0')
        buttons_frame.pack(pady=20)
        
        # Move buttons
        button_style = {"font": ("Arial", 14, "bold"), "width": 10, "height": 2}
        
        rock_btn = tk.Button(
            buttons_frame, 
            text="🪨 Rock", 
            command=lambda: self.make_move("Rock"),
            bg='#ff6b6b', 
            fg='white',
            **button_style
        )
        rock_btn.pack(side=tk.LEFT, padx=10)
        
        paper_btn = tk.Button(
            buttons_frame, 
            text="📄 Paper", 
            command=lambda: self.make_move("Paper"),
            bg='#4ecdc4', 
            fg='white',
            **button_style
        )
        paper_btn.pack(side=tk.LEFT, padx=10)
        
        scissors_btn = tk.Button(
            buttons_frame, 
            text="✂️ Scissors", 
            command=lambda: self.make_move("Scissors"),
            bg='#45b7d1', 
            fg='white',
            **button_style
        )
        scissors_btn.pack(side=tk.LEFT, padx=10)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.results_frame.pack(pady=20, fill='x', padx=20)
        
        # Score frame
        score_frame = tk.Frame(self.root, bg='#f0f0f0')
        score_frame.pack(pady=10)
        
        self.score_label = tk.Label(
            score_frame,
            text="Score - You: 0 | Computer: 0 | Ties: 0",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        self.score_label.pack()
        
        # Game history info
        history_info = tk.Label(
            self.root,
            text=f"Games played: {len(self.game.player_history)} | Model: {self.game.model_type}",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666'
        )
        history_info.pack(pady=5)
        
        # Reset button
        reset_btn = tk.Button(
            self.root,
            text="Reset Game",
            command=self.reset_game,
            font=("Arial", 10),
            bg='#ff4757',
            fg='white',
            padx=20
        )
        reset_btn.pack(pady=10)
    
    def make_move(self, player_move):
        """Handle player move"""
        result = self.game.play_round(player_move)
        self.display_results(result)
        self.update_score()
    
    def display_results(self, result):
        """Display round results"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create results display
        results_text = f"""
🎯 Your Move: {result['player_move']}
🤖 Computer Move: {result['computer_move']}
🔮 AI Predicted: {result['predicted_move']}
🏆 Winner: {result['winner']}
        """
        
        results_label = tk.Label(
            self.results_frame,
            text=results_text,
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#333',
            justify='left'
        )
        results_label.pack()
        
        # Winner highlight
        winner_color = {'Player': '#2ecc71', 'Computer': '#e74c3c', 'Tie': '#f39c12'}
        winner_label = tk.Label(
            self.results_frame,
            text=f"🎉 {result['winner']} wins!" if result['winner'] != 'Tie' else "🤝 It's a tie!",
            font=("Arial", 14, "bold"),
            bg=winner_color[result['winner']],
            fg='white',
            padx=10,
            pady=5
        )
        winner_label.pack(pady=10)
    
    def update_score(self):
        """Update score display"""
        self.score_label.config(
            text=f"Score - You: {self.game.player_score} | Computer: {self.game.computer_score} | Ties: {self.game.ties}"
        )
    
    def reset_game(self):
        """Reset the game"""
        response = messagebox.askyesno(
            "Reset Game", 
            "Are you sure you want to reset the game? This will clear all scores but keep the AI's learning data."
        )
        if response:
            self.game.player_score = 0
            self.game.computer_score = 0
            self.game.ties = 0
            self.update_score()
            
            # Clear results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            messagebox.showinfo("Reset", "Game has been reset!")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def play_terminal_version():
    """Terminal version of the game"""
    print("🤖 Welcome to ML Rock Paper Scissors! 🤖")
    print("The AI learns from your moves to predict your next choice!")
    print("Commands: 'rock', 'paper', 'scissors', 'quit', 'stats'\n")
    
    game = MLRockPaperScissors()
    
    while True:
        player_input = input("Your move (rock/paper/scissors): ").strip().lower()
        
        if player_input == 'quit':
            print("Thanks for playing! 👋")
            break
        elif player_input == 'stats':
            print(f"\n📊 Game Statistics:")
            print(f"Games played: {len(game.player_history)}")
            print(f"Your score: {game.player_score}")
            print(f"Computer score: {game.computer_score}")
            print(f"Ties: {game.ties}")
            print(f"Model trained: {'Yes' if game.is_trained else 'No'}")
            print(f"Model type: {game.model_type}\n")
            continue
        
        # Convert input to proper format
        move_map = {'rock': 'Rock', 'paper': 'Paper', 'scissors': 'Scissors'}
        if player_input not in move_map:
            print("Invalid move! Please enter 'rock', 'paper', or 'scissors'")
            continue
        
        player_move = move_map[player_input]
        result = game.play_round(player_move)
        
        # Display results
        print(f"\n🎯 Your move: {result['player_move']}")
        print(f"🤖 Computer move: {result['computer_move']}")
        print(f"🔮 AI predicted: {result['predicted_move']}")
        print(f"🏆 Winner: {result['winner']}")
        print(f"📊 Score - You: {result['player_score']} | Computer: {result['computer_score']} | Ties: {result['ties']}")
        print("-" * 50)

if __name__ == "__main__":
    print("Choose game mode:")
    print("1. GUI Mode (recommended)")
    print("2. Terminal Mode")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            gui_game = RPSGameGUI()
            gui_game.run()
        except ImportError:
            print("GUI not available. Running terminal mode instead.")
            play_terminal_version()
    else:
        play_terminal_version()