#!/usr/bin/env python3
"""
Quick fix to improve cyberbullying detection accuracy
This will create a better trained model
"""

import pandas as pd
import os
from models.cyberbully_detector import CyberbullyingDetector

# Enhanced training data with more examples
enhanced_data = {
    'text': [
        # Safe examples (30)
        "You're doing great! Keep it up!", "Let's meet for coffee tomorrow", "Great job on your presentation!",
        "Thanks for helping me today", "Have a wonderful day!", "I love spending time with friends",
        "What a beautiful sunny day!", "Thanks for being such a good friend", "You look nice today",
        "Good luck with your exam", "Hope you have a great weekend", "Congratulations on your achievement",
        "Nice work on the project", "See you later", "How are you doing", "That's really cool",
        "I appreciate your help", "You're very talented", "Thanks for listening", "Have a good day",
        "You're awesome", "Great to see you", "Keep up the good work", "You're so helpful",
        "I'm proud of you", "That's amazing", "Well done", "You're incredible", "Love your work", "You're the best",
        
        # Offensive examples (30)
        "I hate you so much, you're worthless", "Nobody likes you, loser", "You're so stupid and ugly",
        "You're pathetic and worthless", "Go kill yourself, nobody cares", "You should just disappear forever",
        "You're a complete failure at everything", "I wish you would just die already", "You're disgusting and pathetic",
        "Everyone thinks you're a joke", "You're such a waste of space", "No one will ever love you",
        "You should be ashamed of yourself", "You're the worst person ever", "I hope you suffer",
        "You deserve to be alone", "You're absolutely useless", "Go away nobody wants you here",
        "You make me sick", "You're a total freak", "Kill yourself", "You're trash",
        "Nobody cares about you", "You're so dumb", "I hate your face", "You're disgusting",
        "Go die", "You suck", "You're annoying", "Shut up loser"
    ],
    'label': [0]*30 + [1]*30  # 30 safe (0) + 30 offensive (1)
}

def main():
    print("🔧 Quick Fix: Improving Cyberbullying Detection")
    print("=" * 50)
    
    # Remove old model
    if os.path.exists('cyberbully_model.h5'):
        os.remove('cyberbully_model.h5')
        print("✅ Removed old model")
    
    if os.path.exists('cyberbully_model_vectorizer.pkl'):
        os.remove('cyberbully_model_vectorizer.pkl')
        print("✅ Removed old vectorizer")
    
    # Create and train new model
    detector = CyberbullyingDetector()
    df = pd.DataFrame(enhanced_data)
    X, y = df['text'], df['label']
    
    print(f"\n📊 Training with {len(X)} examples:")
    print(f"   Safe: {sum(1 for label in y if label == 0)}")
    print(f"   Offensive: {sum(1 for label in y if label == 1)}")
    
    print("\n🏋️ Training model...")
    detector.train(X, y, epochs=200, batch_size=8)
    
    print("\n💾 Saving model...")
    detector.save_model('cyberbully_model')
    
    # Quick test
    print("\n🧪 Testing:")
    test_cases = [
        "You're amazing!",
        "I hate you loser",
        "Have a nice day",
        "Go kill yourself"
    ]
    
    for text in test_cases:
        result = detector.predict(text)
        print(f"   '{text}' -> {result['prediction']} ({result['confidence']:.2f})")
    
    print("\n✅ Quick fix complete! Restart your Flask app.")

if __name__ == "__main__":
    main()