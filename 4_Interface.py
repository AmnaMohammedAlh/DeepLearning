
"""
LifeHarmony Terminal Interface

A command-line application for the LifeHarmony AI recommender system.
This application collects user information, analyzes life balance priorities,
and provides personalized recommendations.
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import datetime

# Try to import the get_recommendations function from deep_learning_model
try:
    from deep_learning_model import get_recommendations
except ImportError:
    # Define a fallback version if the module isn't available
    def get_recommendations(user_features, model_path="harmony_deep_model.h5",
                           recommendations_path="harmony_deep_model_recommendations.pkl",
                           scaler_path="harmony_deep_model_scaler.pkl",
                           threshold=0.3, top_k=10):
        """
        Get recommendations for a user using the deep learning model.
        This is a fallback implementation if the module can't be imported.
        """
        # Load model and recommendations
        model = tf.keras.models.load_model(model_path)
        
        with open(recommendations_path, "rb") as f:
            unique_recommendations = pickle.load(f)
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        # Apply same scaling to age and budget
        user_features = np.array(user_features).reshape(1, -1)
        user_features[:, [0, 3]] = scaler.transform(user_features[:, [0, 3]])
        
        # Get predictions
        predictions = model.predict(user_features)[0]
        
        # Method 1: Threshold-based selection
        recommended_indices = np.where(predictions > threshold)[0]
        
        # Method 2: If too few or too many recommendations, use top-k
        if len(recommended_indices) < 3 or len(recommended_indices) > top_k:
            recommended_indices = np.argsort(predictions)[-top_k:]
        
        # Get recommendations
        recommendations = [unique_recommendations[i] for i in recommended_indices]
        
        return recommendations

# Define constants for the application
DOMAINS = [
    "Career", "Financial", "Spirituality",
    "Health & Fitness", "Personal Development",
    "Family", "Social", "Fun & Recreation"
]

LIFE_FEATURES = ["Career", "Financial", "Spiritual", "Physical", "Intellectual", "Family", "Social", "Fun"]

# Mapping dictionaries for data processing
MARITAL_STATUS_MAPPING = {"Single": 0, "Married": 1, "Divorced": 0, "Widowed": 0}  # Simplifying to binary
OCCUPATION_MAPPING = {"Full-time": 0, "Part-time": 1, "Freelancer": 2, "Student": 3, "Unemployed": 4}
PERSONALITY_MAPPING = {"Extrovert": 0, "Introvert": 1, "Ambivert": 2}
HOBBY_MAPPING = {"Exercise": 0, "Reading": 1, "Writing": 2, "Art": 3, "Socializing": 4}
PRIORITY_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

# Map domains to the features expected by the model
DOMAIN_TO_FEATURE = {
    "Career": "Career",
    "Financial": "Financial",
    "Spirituality": "Spiritual",
    "Health & Fitness": "Physical",
    "Personal Development": "Intellectual",
    "Family": "Family",
    "Social": "Social",
    "Fun & Recreation": "Fun"
}

class LifeHarmonyTerminal:
    """Terminal interface for the LifeHarmony system."""
    
    def __init__(self):
        """Initialize the application."""
        self.user_info = {}
        self.current_ratings = [5] * len(DOMAINS)  # Default to middle value
        self.goal_ratings = [5] * len(DOMAINS)  # Default to middle value
        self.gaps = {}
        self.priorities = {}
        self.recommendations = []
        
        # Check if model files exist
        self.model_files_exist = self._check_model_files()
        
    def _check_model_files(self):
        """Check if the required model files exist."""
        required_files = [
            "harmony_deep_model.h5",
            "harmony_deep_model_recommendations.pkl",
            "harmony_deep_model_scaler.pkl"
        ]
        
        all_exist = True
        for file in required_files:
            if not os.path.exists(file):
                all_exist = False
                break
                
        return all_exist
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, text):
        """Print a formatted header."""
        width = 80
        print("\n" + "=" * width)
        print(text.center(width))
        print("=" * width + "\n")
    
    def get_input(self, prompt, options=None, input_type=None, default=None, range_values=None):
        """
        Get and validate user input.
        
        Parameters:
        - prompt: The question to ask the user
        - options: List of valid options (for multiple choice)
        - input_type: 'int' or 'float' for numeric input
        - default: Default value if user enters nothing
        - range_values: Tuple (min, max) for numeric input range validation
        
        Returns:
        - Validated user input
        """
        while True:
            # Display the prompt with options if applicable
            if options:
                for i, option in enumerate(options, 1):
                    print(f"{i}. {option}")
                
                if default:
                    default_idx = options.index(default) + 1
                    prompt = f"{prompt} [1-{len(options)}, default: {default_idx}]: "
                else:
                    prompt = f"{prompt} [1-{len(options)}]: "
            elif default is not None:
                prompt = f"{prompt} [default: {default}]: "
            else:
                prompt = f"{prompt}: "
            
            # Get user input
            user_input = input(prompt)
            
            # Handle empty input with default
            if user_input == "" and default is not None:
                return default
            
            # Handle different input types
            if options:
                try:
                    choice = int(user_input)
                    if 1 <= choice <= len(options):
                        return options[choice - 1]
                    else:
                        print(f"Please enter a number between 1 and {len(options)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            elif input_type == 'int':
                try:
                    value = int(user_input)
                    if range_values and (value < range_values[0] or value > range_values[1]):
                        print(f"Please enter a number between {range_values[0]} and {range_values[1]}.")
                        continue
                    return value
                except ValueError:
                    print("Please enter a valid integer.")
            
            elif input_type == 'float':
                try:
                    value = float(user_input)
                    if range_values and (value < range_values[0] or value > range_values[1]):
                        print(f"Please enter a number between {range_values[0]} and {range_values[1]}.")
                        continue
                    return value
                except ValueError:
                    print("Please enter a valid number.")
            
            else:
                return user_input
    
    def collect_user_info(self):
        """Collect basic user information."""
        self.clear_screen()
        self.print_header("LifeHarmony: AI Recommender for a Balanced Life")
        print("Welcome to LifeHarmony! Let's gather some information about you to provide personalized recommendations.")
        print("Step 1: General Information\n")
        
        self.user_info["name"] = self.get_input("What is your name")
        self.user_info["age"] = self.get_input("What is your age", input_type='int', range_values=(1, 120), default=25)
        self.user_info["gender"] = self.get_input("What is your gender", options=["Male", "Female", "Other"], default="Male")
        self.user_info["marital_status"] = self.get_input("What is your marital status", 
                                                         options=["Single", "Married", "Divorced", "Widowed"], 
                                                         default="Single")
        self.user_info["occupation"] = self.get_input("What is your occupation", 
                                                     options=["Full-time", "Part-time", "Freelancer", "Student", "Unemployed"], 
                                                     default="Full-time")
        self.user_info["budget"] = self.get_input("What is your monthly budget for self-improvement (in $)", 
                                                 input_type='int', range_values=(0, 10000), default=1000)
        self.user_info["allocated_time"] = self.get_input("How many hours per week can you allocate for self-improvement", 
                                                         input_type='int', range_values=(1, 40), default=10)
        self.user_info["personality"] = self.get_input("What is your personality type", 
                                                      options=["Extrovert", "Introvert", "Ambivert"], 
                                                      default="Ambivert")
        self.user_info["hobby"] = self.get_input("Which hobby resonates with you the most", 
                                                options=["Exercise", "Reading", "Writing", "Art", "Socializing"], 
                                                default="Reading")
        
        # Display summary
        self.clear_screen()
        self.print_header("Summary of Your Information")
        for key, value in self.user_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        input("\nPress Enter to continue to the next step...")
    
    def collect_current_ratings(self):
        """Collect current satisfaction ratings for each domain."""
        self.clear_screen()
        self.print_header("Step 2: Current Satisfaction Levels")
        print("Rate your current satisfaction level in each life domain on a scale of 1-10:")
        print("(1 = Very Dissatisfied, 10 = Very Satisfied)\n")
        
        for i, domain in enumerate(DOMAINS):
            self.current_ratings[i] = self.get_input(f"Current satisfaction with {domain}", 
                                                    input_type='int', range_values=(1, 10), default=5)
        
        # Visualize current ratings with ASCII art
        self.clear_screen()
        self.print_header("Your Current Satisfaction Levels")
        max_width = 40
        for i, domain in enumerate(DOMAINS):
            rating = self.current_ratings[i]
            bar_width = int((rating / 10) * max_width)
            bar = "█" * bar_width + "░" * (max_width - bar_width)
            print(f"{domain.ljust(20)}: {bar} {rating}/10")
        
        input("\nPress Enter to continue to the next step...")
    
    def collect_goal_ratings(self):
        """Collect goal satisfaction ratings for each domain."""
        self.clear_screen()
        self.print_header("Step 3: Goal Satisfaction Levels")
        print("Rate your desired satisfaction level in each life domain on a scale of 1-10:")
        print("(1 = Low Priority, 10 = High Priority)\n")
        
        for i, domain in enumerate(DOMAINS):
            current = self.current_ratings[i]
            self.goal_ratings[i] = self.get_input(f"Goal satisfaction with {domain} (current: {current})", 
                                                 input_type='int', range_values=(current, 10), default=current)
        
        # Visualize goal ratings with ASCII art
        self.clear_screen()
        self.print_header("Your Goal Satisfaction Levels")
        max_width = 40
        for i, domain in enumerate(DOMAINS):
            current = self.current_ratings[i]
            goal = self.goal_ratings[i]
            current_width = int((current / 10) * max_width)
            goal_width = int((goal / 10) * max_width)
            
            # Create a bar showing current and goal
            bar = "▓" * current_width
            if goal_width > current_width:
                bar += "░" * (goal_width - current_width)
            bar += " " * (max_width - goal_width)
            
            print(f"{domain.ljust(20)}: {bar} {current}→{goal}/10")
        
        input("\nPress Enter to continue to the next step...")
    
    def calculate_priorities(self):
        """Calculate gaps and assign priorities to domains."""
        self.clear_screen()
        self.print_header("Step 4: Analyzing Gaps and Priorities")
        
        print("Calculating priorities based on the gap between current and goal satisfaction levels...\n")
        
        # Calculate gaps
        self.gaps = {
            domain: self.goal_ratings[i] - self.current_ratings[i]
            for i, domain in enumerate(DOMAINS)
        }
        
        # Show a loading animation for effect
        for _ in range(3):
            for char in "|/-\\":
                print(f"\rAnalyzing data... {char}", end="", flush=True)
                time.sleep(0.2)
        print("\rAnalysis complete!       ")
        
        # Normalize gaps (divide by max gap to get values between 0 and 1)
        max_gap = max(abs(val) for val in self.gaps.values())
        normalized_gaps = {key: abs(value) / max_gap for key, value in self.gaps.items()} if max_gap != 0 else {key: 0 for key in self.gaps}
        
        # Assign priorities
        if len(set(normalized_gaps.values())) == 1:  # All gaps are equal
            # Tie-breaking rule: Alphabetical order of domain names
            sorted_domains = sorted(normalized_gaps.keys())
            self.priorities = {
                key: "High" if key == sorted_domains[0] else "Medium" if key == sorted_domains[1] else "Low"
                for key in normalized_gaps
            }
        else:
            # Assign priorities based on thresholds
            thresholds = np.percentile(list(normalized_gaps.values()), [33.33, 66.67])
            self.priorities = {
                key: "Low" if normalized_gaps[key] <= thresholds[0] else
                     "Medium" if normalized_gaps[key] <= thresholds[1] else
                     "High"
                for key in normalized_gaps
            }
        
        # Display priorities
        print("\nDomain Priorities:\n")
        
        high_priorities = [domain for domain, priority in self.priorities.items() if priority == "High"]
        medium_priorities = [domain for domain, priority in self.priorities.items() if priority == "Medium"]
        low_priorities = [domain for domain, priority in self.priorities.items() if priority == "Low"]
        
        print("HIGH PRIORITY:".ljust(15), ", ".join(high_priorities))
        print("MEDIUM PRIORITY:".ljust(15), ", ".join(medium_priorities))
        print("LOW PRIORITY:".ljust(15), ", ".join(low_priorities))
        
        input("\nPress Enter to continue to the next step...")
    
    def generate_recommendations(self):
        """Generate personalized recommendations using the deep learning model."""
        self.clear_screen()
        self.print_header("Step 5: Generating Personalized Recommendations")
        
        if not self.model_files_exist:
            print("Warning: Model files not found. Using fallback recommendations.")
            print("To get fully personalized recommendations, ensure the following files exist:")
            print("- harmony_deep_model.h5")
            print("- harmony_deep_model_recommendations.pkl")
            print("- harmony_deep_model_scaler.pkl\n")
            
            # Provide some fallback recommendations based on high priority domains
            self.recommendations = self._generate_fallback_recommendations()
        else:
            print("Generating recommendations from our deep learning model...\n")
            
            # Show a loading animation
            for _ in range(5):
                for char in "|/-\\":
                    print(f"\rProcessing your data... {char}", end="", flush=True)
                    time.sleep(0.2)
            print("\rRecommendations ready!       \n")
            
            # Prepare user's state vector
            user_state_vector = [
                self.user_info["age"],
                MARITAL_STATUS_MAPPING[self.user_info["marital_status"]],
                OCCUPATION_MAPPING[self.user_info["occupation"]],
                self.user_info["budget"],
                PERSONALITY_MAPPING[self.user_info["personality"]],
                HOBBY_MAPPING[self.user_info["hobby"]],
            ]
            
            # Add priorities
            for feature in LIFE_FEATURES:
                matching_domain = None
                for domain in DOMAINS:
                    if domain in DOMAIN_TO_FEATURE and DOMAIN_TO_FEATURE[domain] == feature:
                        matching_domain = domain
                        break
                
                if matching_domain:
                    user_state_vector.append(PRIORITY_MAPPING[self.priorities[matching_domain]])
                else:
                    # Default to Medium if no matching domain
                    user_state_vector.append(PRIORITY_MAPPING["Medium"])
            
            try:
                # Generate recommendations using the deep learning model
                self.recommendations = get_recommendations(
                    user_state_vector,
                    model_path="harmony_deep_model.h5",
                    recommendations_path="harmony_deep_model_recommendations.pkl",
                    scaler_path="harmony_deep_model_scaler.pkl",
                    threshold=0.3,
                    top_k=10
                )
            except Exception as e:
                print(f"Error generating recommendations: {e}")
                print("Using fallback recommendations instead.\n")
                self.recommendations = self._generate_fallback_recommendations()
        
        # Display the recommendations
        if self.recommendations:
            print("Here are your personalized recommendations:\n")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No recommendations could be generated. Please try again.")
        
        input("\nPress Enter to continue to the next step...")
    
    def _generate_fallback_recommendations(self):
        """Generate simple fallback recommendations based on user preferences."""
        recommendations = []
        
        # Get high priority domains
        high_priority_domains = [domain for domain, priority in self.priorities.items() if priority == "High"]
        
        # Add general recommendations for each high priority domain
        for domain in high_priority_domains:
            if domain == "Career":
                recommendations.append("Allocate 1-3 hours a week into improving your career-related skills")
                recommendations.append("Update your resume and professional profiles regularly")
                
            elif domain == "Financial":
                recommendations.append("Start tracking weekly expenses using an app")
                recommendations.append("Set aside a small amount each month for savings")
                
            elif domain == "Spirituality":
                recommendations.append("Allocate time for meditation or reflective practices")
                recommendations.append("Spend time in nature regularly")
                
            elif domain == "Health & Fitness":
                recommendations.append("Incorporate regular physical activity into your routine")
                recommendations.append("Prioritize a balanced diet and adequate sleep")
                
            elif domain == "Personal Development":
                recommendations.append("Set aside time for reading or learning new skills")
                recommendations.append("Join online courses or workshops in areas of interest")
                
            elif domain == "Family":
                recommendations.append("Schedule regular quality time with family members")
                recommendations.append("Create and maintain family traditions or rituals")
                
            elif domain == "Social":
                recommendations.append("Make an effort to connect with friends regularly")
                recommendations.append("Join groups or clubs aligned with your interests")
                
            elif domain == "Fun & Recreation":
                recommendations.append("Schedule dedicated time for activities you enjoy")
                recommendations.append("Try new hobbies or revisit old ones you enjoyed")
        
        # Add personalized recommendations based on hobbies and personality
        hobby = self.user_info["hobby"]
        personality = self.user_info["personality"]
        
        if hobby == "Exercise":
            recommendations.append("Try different types of physical activities to keep your routine fresh")
        elif hobby == "Reading":
            recommendations.append("Join a book club to discuss books and meet like-minded people")
        elif hobby == "Writing":
            recommendations.append("Set aside regular time for writing, even if just for personal reflection")
        elif hobby == "Art":
            recommendations.append("Explore different artistic mediums to expand your creative expression")
        elif hobby == "Socializing":
            recommendations.append("Plan regular gatherings or outings with friends and acquaintances")
        
        if personality == "Introvert":
            recommendations.append("Create a comfortable personal space for recharging and reflection")
        elif personality == "Extrovert":
            recommendations.append("Seek opportunities for group activities and social engagement")
        elif personality == "Ambivert":
            recommendations.append("Balance social activities with personal time for optimal well-being")
        
        # Return a maximum of 10 recommendations
        return recommendations[:10]
    
    def save_results(self):
        """Save recommendations and user profile to a file."""
        self.clear_screen()
        self.print_header("Save Your Results")
        
        save_choice = self.get_input("Would you like to save your results to a file", 
                                    options=["Yes", "No"], default="Yes")
        
        if save_choice == "Yes":
            # Create a directory for saving results if it doesn't exist
            os.makedirs("lifeharmony_results", exist_ok=True)
            
            # Generate a filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lifeharmony_results/{self.user_info['name'].replace(' ', '_')}_{timestamp}.txt"
            
            try:
                with open(filename, 'w') as f:
                    # Write header
                    f.write("=" * 80 + "\n")
                    f.write("LIFEHARMONY PERSONAL RECOMMENDATIONS\n")
                    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Write user information
                    f.write("PERSONAL INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    for key, value in self.user_info.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
                    
                    # Write current and goal ratings
                    f.write("LIFE DOMAIN RATINGS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'Domain'.ljust(20)}{'Current'.center(10)}{'Goal'.center(10)}{'Gap'.center(10)}{'Priority'.center(10)}\n")
                    
                    for i, domain in enumerate(DOMAINS):
                        current = self.current_ratings[i]
                        goal = self.goal_ratings[i]
                        gap = goal - current
                        priority = self.priorities[domain]
                        
                        f.write(f"{domain.ljust(20)}{str(current).center(10)}{str(goal).center(10)}{str(gap).center(10)}{priority.center(10)}\n")
                    f.write("\n")
                    
                    # Write recommendations
                    f.write("PERSONALIZED RECOMMENDATIONS\n")
                    f.write("-" * 80 + "\n")
                    for i, rec in enumerate(self.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    
                    # Write footer
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("Thank you for using LifeHarmony!\n")
                    f.write("=" * 80 + "\n")
                
                print(f"\nResults saved successfully to: {filename}")
            except Exception as e:
                print(f"\nError saving results: {e}")
        
        input("\nPress Enter to continue...")
    
    def display_goodbye(self):
        """Display a goodbye message."""
        self.clear_screen()
        self.print_header("Thank You for Using LifeHarmony!")
        
        print("""
We hope these recommendations help you achieve a more balanced and fulfilling life!
Remember, small consistent steps lead to significant long-term changes.

Best wishes on your journey to a more harmonious life!
        """)
    
    def plot_wheel_of_life(self, save_path=None):
        """
        Create and optionally save a visualization of the Wheel of Life.
        
        Parameters:
        - save_path: Path to save the visualization (optional)
        """
        try:
            # Convert ratings to numpy arrays for easier manipulation
            current_ratings = np.array(self.current_ratings)
            goal_ratings = np.array(self.goal_ratings)
            
            # Number of domains
            N = len(DOMAINS)
            
            # Angles for each domain (divide the plot into equal parts)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            
            # Make the plot circular by repeating the first value
            current_ratings = np.append(current_ratings, current_ratings[0])
            goal_ratings = np.append(goal_ratings, goal_ratings[0])
            angles += angles[:1]
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Plot current ratings
            ax.plot(angles, current_ratings, 'b-', linewidth=2, label='Current')
            ax.fill(angles, current_ratings, 'blue', alpha=0.1)
            
            # Plot goal ratings
            ax.plot(angles, goal_ratings, 'r-', linewidth=2, label='Goal')
            ax.fill(angles, goal_ratings, 'red', alpha=0.1)
            
            # Set the angular ticks
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(DOMAINS)
            
            # Set radial ticks
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'])
            ax.set_ylim(0, 10)
            
            # Add a legend
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add a title
            plt.title('Life Balance Wheel', size=20)
            
            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path)
                print(f"Wheel of Life visualization saved to: {save_path}")
            
            # Display the plot (only works in environments with GUI)
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Visualization could not be created. This feature works best in a GUI environment.")
    
    def run(self):
        """Run the full application flow."""
        try:
            self.collect_user_info()
            self.collect_current_ratings()
            self.collect_goal_ratings()
            self.calculate_priorities()
            self.generate_recommendations()
            
            # Try to create a visualization, but don't halt the program if it fails
            try:
                visualization_choice = self.get_input("Would you like to visualize your Wheel of Life", 
                                                     options=["Yes", "No"], default="Yes")
                
                if visualization_choice == "Yes":
                    # Create directory for visualizations if it doesn't exist
                    os.makedirs("lifeharmony_visualizations", exist_ok=True)
                    
                    # Generate a filename with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_path = f"lifeharmony_visualizations/{self.user_info['name'].replace(' ', '_')}_{timestamp}.png"
                    
                    self.plot_wheel_of_life(save_path=viz_path)
            except:
                print("Visualization could not be created. Continuing with the program...")
            
            self.save_results()
            self.display_goodbye()
            
        except KeyboardInterrupt:
            self.clear_screen()
            print("\nProgram terminated by user. Thank you for using LifeHarmony!")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("The program will now exit.")

if __name__ == "__main__":
    # Create and run the application
    app = LifeHarmonyTerminal()
    app.run()