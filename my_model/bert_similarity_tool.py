# Text Similarity Analysis Tool - Offline Version
# Works without downloading BERT models using TF-IDF and Word2Vec-like approaches

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

class TextSimilarityAnalyzer:
    def __init__(self):
        """Initialize the analyzer with multiple similarity methods"""
        print("✅ Initializing Text Similarity Analyzer (Offline Mode)")
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            max_features=5000
        )
        self.vocabulary = set()
        print("🚀 Ready to analyze text similarity!")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def calculate_jaccard_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        words1 = set(self.preprocess_text(text1).split())
        words2 = set(self.preprocess_text(text2).split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_tfidf_similarity(self, text1, text2):
        """Calculate TF-IDF based cosine similarity"""
        texts = [self.preprocess_text(text1), self.preprocess_text(text2)]
        
        try:
            tfidf_matrix = self.tfidf.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def calculate_word_overlap_similarity(self, text1, text2):
        """Calculate similarity based on word overlap with frequency weighting"""
        words1 = self.preprocess_text(text1).split()
        words2 = self.preprocess_text(text2).split()
        
        if not words1 or not words2:
            return 0.0
        
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # Calculate weighted overlap
        overlap = 0
        for word in counter1:
            if word in counter2:
                overlap += min(counter1[word], counter2[word])
        
        # Normalize by average length
        avg_length = (len(words1) + len(words2)) / 2
        return overlap / avg_length if avg_length > 0 else 0.0
    
    def calculate_composite_similarity(self, text1, text2):
        """Calculate composite similarity using multiple methods"""
        jaccard = self.calculate_jaccard_similarity(text1, text2)
        tfidf = self.calculate_tfidf_similarity(text1, text2)
        word_overlap = self.calculate_word_overlap_similarity(text1, text2)
        
        # Weighted average (TF-IDF gets highest weight)
        composite = (0.5 * tfidf + 0.3 * jaccard + 0.2 * word_overlap)
        
        return {
            'composite': composite,
            'jaccard': jaccard,
            'tfidf': tfidf,
            'word_overlap': word_overlap
        }
    
    def find_best_match(self, query, candidates, method='composite'):
        """Find the best matching candidate for a query"""
        if not query or not candidates:
            return None, 0.0, []
        
        similarities = []
        for candidate in candidates:
            if method == 'composite':
                result = self.calculate_composite_similarity(query, candidate)
                similarity = result['composite']
            elif method == 'jaccard':
                similarity = self.calculate_jaccard_similarity(query, candidate)
            elif method == 'tfidf':
                similarity = self.calculate_tfidf_similarity(query, candidate)
            else:
                similarity = self.calculate_word_overlap_similarity(query, candidate)
            
            similarities.append(similarity)
        
        if not similarities:
            return None, 0.0, []
        
        best_index = similarities.index(max(similarities))
        best_match = candidates[best_index]
        best_score = similarities[best_index]
        
        return best_match, best_score, similarities
    
    def classify_relationship(self, entity1, entity2, threshold=0.3, method='composite'):
        """Classify if two entities are related"""
        if method == 'composite':
            result = self.calculate_composite_similarity(entity1, entity2)
            similarity = result['composite']
        else:
            similarity = self.calculate_tfidf_similarity(entity1, entity2)
        
        is_match = similarity >= threshold
        return is_match, similarity

def main():
    """Main function to run all examples"""
    try:
        # Initialize the analyzer
        print("="*70)
        print("🔍 TEXT SIMILARITY ANALYSIS TOOL - OFFLINE VERSION")
        print("="*70)
        
        analyzer = TextSimilarityAnalyzer()
        
        # ============================================================================
        # Example 1: FAQ Matching System
        # ============================================================================
        print("\n" + "="*60)
        print("❓ FAQ MATCHING SYSTEM")
        print("="*60)

        # FAQ database
        faq_questions = [
            "How can I reset my password?",
            "Where is the library located?",
            "What is Artificial Intelligence?",
            "How to apply for a scholarship?",
            "What are the cafeteria opening hours?",
            "How do I register for courses?",
            "What is machine learning?",
            "Where can I find study materials?",
            "How to change my email address?",
            "What is deep learning?"
        ]

        # Test queries
        test_queries = [
            "what is AI",
            "password reset help", 
            "library location",
            "course registration process",
            "ML explanation needed",
            "change email",
            "study resources"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            best_match, score, all_scores = analyzer.find_best_match(query, faq_questions)
            
            if best_match:
                print(f"📋 Best Match: '{best_match}'")
                print(f"📊 Similarity Score: {score:.3f}")
                
                # Show detailed similarity breakdown
                result = analyzer.calculate_composite_similarity(query, best_match)
                print(f"   📈 Breakdown - TF-IDF: {result['tfidf']:.3f}, "
                      f"Jaccard: {result['jaccard']:.3f}, "
                      f"Word Overlap: {result['word_overlap']:.3f}")
                
                # Show top 3 matches
                if all_scores:
                    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
                    print("🏆 Top 3 matches:")
                    for i in range(min(3, len(sorted_indices))):
                        idx = sorted_indices[i]
                        print(f"   {i+1}. {faq_questions[idx]} (Score: {all_scores[idx]:.3f})")

        # ============================================================================
        # Example 2: Entity-Relationship Classification
        # ============================================================================
        print("\n" + "="*60)
        print("🏏 ENTITY-RELATIONSHIP CLASSIFICATION")
        print("="*60)

        # Test cases for relationship classification
        test_relationships = [
            ("MS Dhoni", "India cricket captain"),
            ("MS Dhoni", "Australian cricketer"),
            ("Ricky Ponting", "Australia cricket"),
            ("Virat Kohli", "Indian batsman"),
            ("Kane Williamson", "New Zealand cricket"),
            ("Joe Root", "Indian player"),  # Wrong pairing
            ("Lionel Messi", "Argentina football"),
            ("Cristiano Ronaldo", "Brazilian football"),  # Wrong pairing
            ("Python programming", "coding language"),
            ("Snake animal", "programming language"),  # Different context
        ]

        threshold = 0.25
        print(f"🎯 Classification threshold: {threshold}")
        print("\n📊 Results:")

        for entity, context in test_relationships:
            is_match, similarity = analyzer.classify_relationship(entity, context, threshold)
            status = "✅ MATCH" if is_match else "❌ NO MATCH"
            print(f"{entity:20} - {context:25} | Score: {similarity:.3f} | {status}")

        # ============================================================================
        # Example 3: Semantic Text Analysis
        # ============================================================================
        print("\n" + "="*60)
        print("🔬 SEMANTIC TEXT ANALYSIS")
        print("="*60)

        # Compare semantic similarity between different sentence pairs
        sentence_pairs = [
            ("The cat sits on the mat", "A feline rests on the carpet"),
            ("I love programming", "Coding is my passion"),
            ("The weather is sunny", "It's raining heavily"),
            ("Artificial Intelligence", "Machine Learning"),
            ("Python programming language", "Snake in the garden"),
            ("University library", "Academic resource center"),
            ("Reset my password", "Change my login credentials"),
            ("Book a table", "Reserve a seat"),
        ]

        print("🔍 Semantic similarity analysis:")
        for sent1, sent2 in sentence_pairs:
            result = analyzer.calculate_composite_similarity(sent1, sent2)
            similarity = result['composite']
            
            print(f"\n📝 '{sent1}'")
            print(f"📝 '{sent2}'")
            print(f"📊 Composite Score: {similarity:.3f}", end=" ")
            print(f"(TF-IDF: {result['tfidf']:.2f}, Jaccard: {result['jaccard']:.2f})")
            
            if similarity > 0.6:
                print("   → 🟢 Very similar meaning")
            elif similarity > 0.4:
                print("   → 🟡 Moderately similar")
            elif similarity > 0.2:
                print("   → 🟠 Somewhat related")
            else:
                print("   → 🔴 Different meanings")

        # ============================================================================
        # Example 4: Document Similarity Matrix
        # ============================================================================
        print("\n" + "="*60)
        print("📄 DOCUMENT SIMILARITY MATRIX")
        print("="*60)

        documents = [
            "Machine learning algorithms learn from data automatically",
            "Deep learning uses neural networks with multiple layers", 
            "Python is a popular programming language for development",
            "Data science involves extracting insights from large datasets",
            "JavaScript is primarily used for web development projects",
            "Artificial intelligence simulates human cognitive processes"
        ]

        print("📋 Documents:")
        for i, doc in enumerate(documents):
            print(f"   {i}: {doc}")

        print(f"\n🔗 Most similar document pairs:")
        similarities = []
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                result = analyzer.calculate_composite_similarity(documents[i], documents[j])
                similarities.append((i, j, result['composite']))

        similarities.sort(key=lambda x: x[2], reverse=True)
        for i, (doc1_idx, doc2_idx, sim) in enumerate(similarities[:5]):
            print(f"\n   {i+1}. Documents {doc1_idx} & {doc2_idx}: {sim:.3f}")
            print(f"      '{documents[doc1_idx][:50]}...'")
            print(f"      '{documents[doc2_idx][:50]}...'")

        # ============================================================================
        # Method Comparison
        # ============================================================================
        print("\n" + "="*60)
        print("⚖️  SIMILARITY METHOD COMPARISON")
        print("="*60)
        
        test_pair = ("artificial intelligence", "machine learning")
        result = analyzer.calculate_composite_similarity(test_pair[0], test_pair[1])
        
        print(f"📝 Comparing: '{test_pair[0]}' vs '{test_pair[1]}'")
        print(f"📊 Results:")
        print(f"   • TF-IDF Similarity:    {result['tfidf']:.4f}")
        print(f"   • Jaccard Similarity:   {result['jaccard']:.4f}")  
        print(f"   • Word Overlap:         {result['word_overlap']:.4f}")
        print(f"   • Composite Score:      {result['composite']:.4f}")

        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("💡 This offline version uses TF-IDF, Jaccard, and word overlap methods")
        print("📝 For production use, consider using BERT or other transformer models")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()