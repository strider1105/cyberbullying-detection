#!/usr/bin/env python3
"""
Script to retrain the cyberbullying detection model with better data
Run this to improve the model's accuracy
"""

import pandas as pd
import os
from models.cyberbully_detector import CyberbullyingDetector

def get_enhanced_training_data():
    """Get enhanced training data with more examples for better detection"""
    return {
        'text': [
            # Safe/Non-offensive examples (0)
            "You're doing great! Keep it up!",
            "Let's meet for coffee tomorrow",
            "Great job on your presentation!",
            "Thanks for helping me today",
            "Have a wonderful day!",
            "I love spending time with friends",
            "What a beautiful sunny day!",
            "Thanks for being such a good friend",
            "You look nice today",
            "Good luck with your exam",
            "Hope you have a great weekend",
            "Congratulations on your achievement",
            "Nice work on the project",
            "See you later",
            "How are you doing",
            "That's really cool",
            "I appreciate your help",
            "You're very talented",
            "Thanks for listening",
            "Have a good day",
            "You're awesome",
            "Great to see you",
            "Keep up the good work",
            "You're so helpful",
            "I'm proud of you",
            "That's amazing",
            "Well done",
            "You're incredible",
            "Love your work",
            "You're the best",
            
            # Offensive/Cyberbullying examples (1)
            "I hate you so much, you're worthless",
            "Nobody likes you, loser",
            "You're so stupid and ugly",
            "You're pathetic and worthless",
            "Go kill yourself, nobody cares",
            "You should just disappear forever",
            "You're a complete failure at everything",
            "I wish you would just die already",
            "You're disgusting and pathetic",
            "Everyone thinks you're a joke",
            "You're such a waste of space",
            "No one will ever love you",
            "You should be ashamed of yourself",
            "You're the worst person ever",
            "I hope you suffer",
            "You deserve to be alone",
            "You're absolutely useless",
            "Go away nobody wants you here",
            "You make me sick",
            "You're a total freak",
            "Kill yourself",
            "You're trash",
            "Nobody cares about you",
            "You're so dumb",
            "I hate your face",
            "You're disgusting",
            "Go die",
            "You suck",
            "You're annoying",
            "Shut up loser",
            "You're a piece of garbage",
            "I hope you get hurt",
            "You're so fat and ugly",
            "Everyone hates you",
            "You should kill yourself",
            "You're a complete idiot",
            "I wish you were dead",
            "You're absolutely horrible",
            "You deserve nothing",
            "You're a total waste"
        ],
        'label': [
            # Labels: 0 = safe, 1 = offensive
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 30 Safe examples
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # 40 Offensive examples
        ]
    }

def main():
    print("🚀 Retraining Cyberbullying Detection Model")
    print("=" * 50)
    
    # Remove existing model if it exists
    model_files = ['cyberbully_model.h5', 'cyberbully_model_vectorizer.pkl']
    for file in model_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed old model file: {file}")
    
    # Create detector
    detector = CyberbullyingDetector()
    
    # Get enhanced training data
    print("\n📊 Loading enhanced training data...")
    sample_data = get_enhanced_training_data()
    df_sample = pd.DataFrame(sample_data)
    X, y = df_sample['text'], df_sample['label']
    
    print(f"Total examples: {len(X)}")
    print(f"Safe examples: {sum(1 for label in y if label == 0)}")
    print(f"Offensive examples: {sum(1 for label in y if label == 1)}")
    
    # Train the model
    print("\n🏋️ Training improved model...")
    detector.train(X, y, epochs=150, batch_size=8)
    
    # Save the model
    print("\n💾 Saving improved model...")
    detector.save_model('cyberbully_model')
    
    # Test the model
    print("\n🧪 Testing improved model...")
    test_cases = [
        ("You're amazing and I love you!", "Safe"),
        ("I hate you, you're worthless", "Offensive"),
        ("Have a great day!", "Safe"),
        ("Go kill yourself loser", "Offensive"),
        ("Thanks for your help", "Safe"),
        ("You're so stupid and ugly", "Offensive")
    ]
    
    for text, expected in test_cases:
        result = detector.predict(text)
        status = "✅" if result['prediction'] == expected else "❌"
        print(f"{status} '{text}' -> {result['prediction']} (confidence: {result['confidence']:.2f})")
    
    print("\n🎉 Model retraining complete!")
    print("Restart your Flask app to use the improved model.")

if __name__ == "__main__":
    main()